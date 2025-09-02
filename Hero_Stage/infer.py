import torch
from PIL import Image
from train.src.pipeline import FluxPipeline
from train.src.transformer_flux import FluxTransformer2DModel
from train.src.lora_helper import set_single_lora, set_multi_lora
import os
import json
import numpy as np
from utils import create_video_from_images, plot_loss_curve, \
    cal_psnr, cal_lpips, cal_psnr_mask, cal_clip_similarity, cal_clip_similarity_mask, cal_ssim

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Initialize model
device = "cuda"
base_path = "./_pretrained_model/FLUX.1-dev"  # Path to your base model

pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
transformer = FluxTransformer2DModel.from_pretrained(
    base_path, 
    subfolder="transformer",
    torch_dtype=torch.bfloat16, 
    device=device
)
pipe.transformer = transformer
pipe.to(device)

eval_json_path_ls = [
    "./_input/_json/hero_stage/car_turn/rainbow_2cond_infer.json",
]
for eval_json_path in eval_json_path_ls:
    # eval_json_path = "/mnt/workspace/tongtong.tt/editing/EasyControl/train/examples/enhance_json/10videos_color_input/cat_dog_input/color1_2cond_infer.json"
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    for item in data:
        if item["active"] == False:
            continue

        lora_folder = item["lora_folder"]
        lora_step = item["lora_step"]
        
        image_size = item["image_size"]

        # lora_step = 1 # cia gudie=3.5, contrast=1.0
        # lora_step = 4 # i2vedit 300 steps

        # lora_step = 2 # cia gudie=3.5, contrast=1.67
        # lora_step = 3 # i2vedit 200 steps
        # lora_step = 5 # colormnet

        # lora_step = 11 # anyv2v 0.5_50steps
        # lora_step = 12 # anyv2v 0.5_100steps
        # lora_step = 13 # anyv2v 1.0_50steps

        if lora_step in list(range(50)): # baseline CiA, arbitrary ckpt to fill
            lora_path = f"{lora_folder}/ckpt/checkpoint-400/lora.safetensors"
        else:
            lora_path = f"{lora_folder}/ckpt/checkpoint-{lora_step}/lora.safetensors"
        if "2cond" in eval_json_path:
            mask_folder = item["source_tgt_folder"].replace("content", "mask")
        else:
            mask_folder = item["source_folder"].replace("content", "mask")

        if "2cond" in eval_json_path:
            # set_single_lora(pipe.transformer, lora_path, lora_weights=[1, 1], cond_size=image_size[0])
            set_single_lora(pipe.transformer, lora_path, lora_weights=[1, 1], cond_size=image_size[0])

            source_zero_folder = item["source_zero_folder"]
            source_tgt_folder = item["source_tgt_folder"]
            target_folder = item["target_folder"]
        else:
            set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=image_size[0])
            source_folder = item["source_folder"]
            target_folder = item["target_folder"]


        exp_name = item["exp_name"]
        prompt = item["prompt"]
        appearance_name = item["appearance_name"]
        
        frame_id = item["frame_id"]
        mask_setting = item["mask_setting"]

        inference_output_path = f"{lora_folder}/inference/video_output"

        if mask_setting == "yes":
            inference_eval_path = f"{lora_folder}/inference/eval_metric"
        else:
            inference_eval_path = f"{lora_folder}/inference/eval_metric_womask"

        os.makedirs(f"{inference_output_path}/step{lora_step}/{exp_name}_{appearance_name}", exist_ok=True)
        os.makedirs(f"{inference_eval_path}/{exp_name}_{appearance_name}", exist_ok=True)
        
        psnr_ls = []
        lpips_ls = []
        clip_ls = []
        ssim_ls = []
        
        for i in range(frame_id[0], frame_id[1], frame_id[2]):
            if mask_setting == "yes":
                mask = Image.open(f"{mask_folder}/frame_{i:04d}.png").convert("RGB")
            if target_folder != "":
                try:
                    if "style" in target_folder:
                        gt_image = Image.open(f"{target_folder}/frame_{i:04d}_{appearance_name}.png").convert("RGB")
                    else:
                        gt_image = Image.open(f"{target_folder}/frame_{i:04d}.png").convert("RGB")
                except:
                    gt_image = None
            else:
                gt_image = None
            
            if "2cond" in eval_json_path:
                spatial_image_zero = Image.open(f"{source_zero_folder}/frame_{i:04d}.png").convert("RGB")
                spatial_image_tgt = Image.open(f"{source_tgt_folder}/frame_{i:04d}.png").convert("RGB")
                image = pipe(
                    prompt,
                    height=image_size[0],
                    width=image_size[1],
                    guidance_scale=3.5,
                    num_inference_steps=25,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(5),
                    spatial_images=[spatial_image_zero, spatial_image_tgt],
                    subject_images=[],
                    cond_size=image_size[0],
                ).images[0]
            else:
                spatial_image = Image.open(f"{source_folder}/frame_{i:04d}.png").convert("RGB")
                image = pipe(
                    prompt,
                    height=image_size[0],
                    width=image_size[1],
                    guidance_scale=3.5,
                    num_inference_steps=25,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(5),
                    spatial_images=[spatial_image],
                    subject_images=[],
                    cond_size=image_size[0],
                ).images[0]
        
            if gt_image != None:
                if mask_setting == "yes":
                    psnr = cal_psnr_mask(image, gt_image, mask)
                else:
                    psnr = cal_psnr(image, gt_image)
                    lpips = cal_lpips(image, gt_image)
                    ssim = cal_ssim(image, gt_image)

                # clip_sim = cal_clip_similarity_mask(image, gt_image, mask)
            
                psnr_ls.append(psnr)
                lpips_ls.append(lpips)
                ssim_ls.append(ssim)
                # clip_ls.append(clip_sim)
                with open(f"{inference_eval_path}/{exp_name}_{appearance_name}/step{lora_step}.txt", "a+") as f:
                    # f.write(f"PSNR frame {i}, {psnr}\n") 
                    # f.write(f"lpips frame {i}, {lpips}\n") 
                    # f.write(f"ssim frame {i}, {ssim}\n") 
                    pass

            image.save(f"{inference_output_path}/step{lora_step}/{exp_name}_{appearance_name}/frame_{i:04d}.png")
            print(f"save at: {inference_output_path}/step{lora_step}/{exp_name}_{appearance_name}/frame_{i:04d}.png")

        
            create_video_from_images(
                f"{inference_output_path}/step{lora_step}/{exp_name}_{appearance_name}",
                f"{inference_output_path}/step{lora_step}/{exp_name}_{appearance_name}/video.mp4",
                fps=8,
            )

        # print(f"lora step {lora_step}: ", np.mean(np.array(psnr_ls)))

        mean_PSNR = np.mean(np.array(psnr_ls)).round(4)
        std_PSNR = np.std(np.array(psnr_ls))
        min_PSNR = np.min(np.array(psnr_ls))

        mean_clip = np.mean(np.array(clip_ls))

        with open(f"{inference_eval_path}/{exp_name}_{appearance_name}/step{lora_step}.txt", "a+") as f:
            f.write(f"mean PSNR, {mean_PSNR}\n")
            # f.write(f"mean clip, {mean_clip}\n")

            # f.write(f"std PSNR, {std_PSNR}\n")
            # f.write(f"min PSNR, {min_PSNR}\n")
            # f.write(f"PSNR list: {psnr_ls}\n")
            
            f.write("**************************************\n")

        mean_lpips = np.mean(np.array(lpips_ls)).round(4)
        std_lpips = np.std(np.array(lpips_ls))
        min_lpips = np.min(np.array(lpips_ls))


        with open(f"{inference_eval_path}/{exp_name}_{appearance_name}/step{lora_step}.txt", "a+") as f:
            f.write(f"mean lpips, {mean_lpips}\n")
            # f.write(f"mean clip, {mean_clip}\n")

            # f.write(f"std lpips, {std_lpips}\n")
            # f.write(f"min lpips, {min_lpips}\n")
            # f.write(f"lpips list: {lpips_ls}\n")
            
            f.write("**************************************\n")

        mean_ssim = np.mean(np.array(ssim_ls)).round(4)
        std_ssim = np.std(np.array(ssim_ls))
        min_ssim = np.min(np.array(ssim_ls))


        with open(f"{inference_eval_path}/{exp_name}_{appearance_name}/step{lora_step}.txt", "a+") as f:
            f.write(f"mean ssim, {mean_ssim}\n")
            # f.write(f"mean clip, {mean_clip}\n")

            # f.write(f"std ssim, {std_ssim}\n")
            # f.write(f"min ssim, {min_ssim}\n")
            # f.write(f"ssim list: {ssim_ls}\n")
            
            f.write("**************************************\n")