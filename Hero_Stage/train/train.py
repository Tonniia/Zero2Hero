import argparse
import copy
import logging
import math
import time
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
from contextlib import nullcontext
from pathlib import Path
import re
from safetensors.torch import save_file

from PIL import Image
import numpy as np
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    convert_unet_state_dict_to_peft
)

from src.prompt_helper import *
from src.lora_helper import *
from src.pipeline import FluxPipeline, resize_position_encoding, prepare_latent_subject_ids

from src.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
# from src.layers_ import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor

from src.transformer_flux import FluxTransformer2DModel
from src.jsonl_datasets import make_train_dataset, collate_fn

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

def cal_psnr(imageA, imageB):
    target_size = (512, 512)
    imageA = np.array(imageA.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)
    imageB = np.array(imageB.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)
    mse_value = np.mean((imageA-imageB) ** 2)
    if mse_value == 0:
        return 100
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value

def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        step,
        torch_dtype,
        is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    # for tracker in accelerator.trackers:
    #     phase_name = "test" if is_final_validation else "validation"
    #     if tracker.name == "tensorboard":
    #         np_images = np.stack([np.asarray(img) for img in images])
    #         tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
    #     if tracker.name == "wandb":
    #         tracker.log(
    #             {
    #                 phase_name: [
    #                     wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
    #                 ]
    #             }
    #         )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--lora_num", type=int, default=2, help="number of the lora.")
    parser.add_argument("--cond_size", type=int, default=512, help="size of the condition data.")
    parser.add_argument("--noise_size", type=int, default=1024, help="max side of the training data.")
    parser.add_argument("--test_h", type=int, default=1024, help="max side of the training data.")
    parser.add_argument("--test_w", type=int, default=1024, help="max side of the training data.")
    parser.add_argument("--mode",type=str,default=None,help="The mode of the controller. Choose between ['depth', 'pose', 'canny'].")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./_input/_json/hero_stage/car_turn/rainbow_2cond_1pairs.jsonl",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./_pretrained_model/FLUX.1-dev",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--spatial_column",
        type=str,
        default="source_zero,source_tgt",
        # default="source",
        help="The column of the dataset containing the canny image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--subject_column",
        type=str,
        default="None",
        help="The column of the dataset containing the subject image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="target",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a car is on the road.",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--subject_test_images",
        type=str,
        nargs="+",
        default=["/tiamat-NAS/zhangyuxuan/datasets/benchmark_dataset/decoritems_woodenpot/0.png"],
        help="A list of subject test image paths.",
    )
    parser.add_argument(
        "--eval_json_path",
        type=str,
        help="eval path.",
    )
    parser.add_argument(
        "--spatial_test_images",
        type=str,
        nargs="+",
        default=[],
        help="A list of spatial test image paths.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20,
        help=(
            "Run validation every X epochs. validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[128, 128],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alphas",
        type=int,
        nargs="+",
        default=[128, 128],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./_result/tmp/Mode2/car_turn_rainbow",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
             "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
             "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    t0 = time.time()
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    run_name = time.strftime("%Y%m%d-%H%M%S")
    args.output_dir = f"{args.output_dir}/{run_name}"
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_dir = f"{args.output_dir}/ckpt"
        config_dir = f"{args.output_dir}/configs"
        valid_dir = f"{args.output_dir}/output"
        logging_dir = config_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        if "," in args.spatial_column:
            shutil.copy("./Hero_Stage/train/train_spatial_2cond.sh", f"{config_dir}/train_backup.sh")
        else:
            shutil.copy("./Hero_Stage/train/train_spatial_1cond.sh", f"{config_dir}/train_backup.sh")
        shutil.copy(args.train_data_dir, f"{config_dir}/train_json.jsonl")

    if args.subject_column == "None":
        args.subject_column = None
    if args.spatial_column == "None":
        args.spatial_column = None

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    #### lora_layers ####
    if args.pretrained_lora_path is not None:
        lora_path = args.pretrained_lora_path
        checkpoint = load_checkpoint(lora_path)
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        number = 1
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                            lora_state_dicts[key] = value
                
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.down.weight', None)
                    lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.up.weight', None)
                
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                            lora_state_dicts[key] = value
                
                print("setting LoRA Processor for", name)        
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()
    else:
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
            else:
                lora_attn_procs[name] = attn_processor        
    ######################
    transformer.set_attn_processor(lora_attn_procs)
    transformer.train()
    for n, param in transformer.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'M parameters')

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        global_step = int(path.split("-")[-1])
        initial_global_step = global_step
    else:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    transformer_parameters_with_lr = {"params": params_to_optimize, "lr": args.learning_rate}
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        [transformer_parameters_with_lr],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    # Dataset and DataLoaders creation:
    train_dataset = make_train_dataset(args, tokenizers, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.resume_from_checkpoint:
        first_epoch = global_step // num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_name = "Easy_Control"
    #     accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # some fixed parameters 
    vae_scale_factor = 16
    height_cond = 2 * (args.cond_size // vae_scale_factor)
    width_cond = 2 * (args.cond_size // vae_scale_factor)        
    offset = 64
        
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                
                tokens = [batch["text_ids_1"], batch["text_ids_2"]]
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_token_ids(text_encoders, tokens, accelerator)
                prompt_embeds = prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
                text_ids = text_ids.to(dtype=vae.dtype, device=accelerator.device)
                
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                height_ = 2 * (int(pixel_values.shape[-2]) // vae_scale_factor)
                width_ = 2 * (int(pixel_values.shape[-1]) // vae_scale_factor)

                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                latent_image_ids, cond_latent_image_ids = resize_position_encoding(
                    model_input.shape[0],
                    height_,
                    width_,
                    height_cond,
                    width_cond,
                    accelerator.device,
                    weight_dtype,
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                
                latent_image_ids_to_concat = [latent_image_ids]
                packed_cond_model_input_to_concat = []
                
                if args.subject_column is not None:
                    subject_pixel_values = batch["subject_pixel_values"].to(dtype=vae.dtype)
                    subject_input = vae.encode(subject_pixel_values).latent_dist.sample()
                    subject_input = (subject_input - vae_config_shift_factor) * vae_config_scaling_factor
                    subject_input = subject_input.to(dtype=weight_dtype)             
                    sub_number = subject_pixel_values.shape[-2] // args.cond_size
                    latent_subject_ids = prepare_latent_subject_ids(height_cond, width_cond, accelerator.device, weight_dtype)
                    latent_subject_ids[:, 1] += offset
                    sub_latent_image_ids = torch.concat([latent_subject_ids for _ in range(sub_number)], dim=-2)
                    latent_image_ids_to_concat.append(sub_latent_image_ids)
                    
                    packed_subject_model_input = FluxPipeline._pack_latents(    
                        subject_input,
                        batch_size=subject_input.shape[0],
                        num_channels_latents=subject_input.shape[1],
                        height=subject_input.shape[2],
                        width=subject_input.shape[3],
                    )
                    packed_cond_model_input_to_concat.append(packed_subject_model_input)
                
                if args.spatial_column is not None:
                    cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)             
                    cond_input = vae.encode(cond_pixel_values).latent_dist.sample()
                    cond_input = (cond_input - vae_config_shift_factor) * vae_config_scaling_factor
                    cond_input = cond_input.to(dtype=weight_dtype)
                    cond_number = cond_pixel_values.shape[-2] // args.cond_size
                    cond_latent_image_ids = torch.concat([cond_latent_image_ids for _ in range(cond_number)], dim=-2)
                    latent_image_ids_to_concat.append(cond_latent_image_ids)

                    packed_cond_model_input = FluxPipeline._pack_latents(
                        cond_input,
                        batch_size=cond_input.shape[0],
                        num_channels_latents=cond_input.shape[1],
                        height=cond_input.shape[2],
                        width=cond_input.shape[3],
                    )
                    packed_cond_model_input_to_concat.append(packed_cond_model_input)
                    
                latent_image_ids = torch.concat(latent_image_ids_to_concat, dim=-2)
                cond_packed_noisy_model_input = torch.concat(packed_cond_model_input_to_concat, dim=-2)

                # handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    cond_hidden_states=cond_packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(pixel_values.shape[-2]),
                    width=int(pixel_values.shape[-1]),
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )

                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (transformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(ckpt_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(ckpt_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(ckpt_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict()
                        lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    pipeline = FluxPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        transformer=accelerator.unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    with open(args.eval_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f) 
                        spatial_images_ls = []
                        for idx, item in enumerate(data):
                            if item["active"] == False:
                                continue
                            if "_2" not in args.eval_json_path:
                                source_path = item.get("source_path")
                                target_path = item.get("target_path")
                                position_delta = item.get("position_delta")
                                text = item.get("text")
                                if item["is_metric"] == True:
                                    gt_image = Image.open(target_path).convert("RGB")
                                else:
                                    gt_image = None
                                spatial_images_ls = [Image.open(source_path).convert("RGB")]

                                pipeline_args = {"prompt": text,
                                                "spatial_images": spatial_images_ls,
                                                "subject_images": [],
                                                "height": args.test_h,
                                                "width": args.test_w,
                                                "cond_size": args.cond_size,
                                                "guidance_scale": 3,
                                                "num_inference_steps": 20,
                                                "max_sequence_length": 512,
                                                }
                            else:
                                source_zero = item.get("source_zero")
                                source_tgt = item.get("source_tgt")
                                target_path = item.get("target_path")
                                position_delta = item.get("position_delta")
                                text = item.get("text")
                                if item["is_metric"] == True:
                                    gt_image = Image.open(target_path).convert("RGB")
                                else:
                                    gt_image = None
                                spatial_images_ls = [Image.open(source_zero).convert("RGB"), Image.open(source_tgt).convert("RGB")]

                                pipeline_args = {"prompt": text,
                                                "spatial_images": spatial_images_ls,
                                                "subject_images": [],
                                                "height": args.test_h,
                                                "width": args.test_w,
                                                "cond_size": args.cond_size,
                                                "guidance_scale": 3,
                                                "num_inference_steps": 20,
                                                "max_sequence_length": 512,
                                                }                               
                            
                            images = log_validation(
                                pipeline=pipeline,
                                args=args,
                                accelerator=accelerator,
                                pipeline_args=pipeline_args,
                                step=global_step,
                                torch_dtype=weight_dtype,
                            )
                            save_path = valid_dir
                            # os.makedirs(save_path, exist_ok=True)
                            # save_folder = os.path.join(save_path, f"checkpoint-{global_step}")
                            # os.makedirs(save_folder, exist_ok=True)
                            # for i, img in enumerate(images):
                            if gt_image != None:
                                gt_image.save(os.path.join(save_path, f"_gt_enhance_{idx}.jpg"))

                                psnr = cal_psnr(gt_image, images[0])
                                

                                with open(f"{save_path}/__eval_log__.txt", "a+") as f:
                                    f.write(f"PSNR step-{global_step}: {psnr}, total_time: {time.time()-t0}\n")


                            images[0].save(os.path.join(save_path, f"lora_{global_step}_enhance_{idx}.jpg"))
                        del pipeline

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)