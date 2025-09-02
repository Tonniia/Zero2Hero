import torch
import numpy as np
import copy
import os
import itertools
import cv2
from tqdm import tqdm
from time import time
from config import get_args
import json
from einops import rearrange

from stable_diffusion import load_stable_diffusion, encode_latent, decode_latent, get_text_embedding, get_unet_layers, attention_op  # load SD
from extract_dift_retrieval import *
from utils import * 

def image_inversion(image_path, text, unet_wrapper):
    print("style path: ", image_path)
    image = cv2.imread(image_path)[:, :, ::-1]
    image = cv2.resize(image, (img_size[1], img_size[0]))
    denoise_kwargs = unet_wrapper.get_text_condition(text)

    unet_wrapper.trigger_get_qkv = True
    unet_wrapper.trigger_modify_qkv = False
    
    latent = encode_latent(normalize(image).to(device=vae.device, dtype=dtype), vae)
    
    print(f"Invert: {image_path}...")
    images, latents = unet_wrapper.invert_process(latent, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
    latent = latents[-1]
    
    # ================= IMPORTANT =================
    # save key value from style image
    features = copy.deepcopy(unet_wrapper.attn_features)
    # =============================================
    return latent, features, image

def iter_combine(gamma, tau, k):
    gamma_ls = gamma
    tau_ls = tau
    top_k_ls = k

    fixed_params = {
        "injection_layers": list(range(3, 12)),
    }
    combinations = list(itertools.product(gamma_ls, tau_ls, top_k_ls))
    params_list = []
    for gamma, tau, top_k in combinations:
        params = fixed_params.copy()
        params.update({
            "gamma": gamma,
            "tau": tau,
            "top_k": top_k,
        })
        params_list.append(params) 
    return params_list

# class for obtain and override the features
class style_transfer_module():        
    def __init__(self,
        unet, vae, text_encoder, tokenizer, scheduler, style_transfer_params = None,
    ):  
        style_transfer_params_default = {
            'gamma': None,
            'tau': None,
            'injection_layers': None
        }
        if style_transfer_params is not None:
            style_transfer_params_default.update(style_transfer_params)
        self.style_transfer_params = style_transfer_params_default
        
        self.unet = unet # SD unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.attn_features = {} # where to save key value (attention block feature)
        self.attn_features_modify = {} # where to save key value to modify (attention block feature)
        self.cur_t = None
        
        # Get residual and attention block in decoder
        # [0 ~ 11], total 12 layers
        resnet, attn = get_unet_layers(unet)
        
        # where to inject key and value
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']
    
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value("layer{}_attn".format(i)))
        
        # Modify hook (if you change query key value)
        for i in qkv_injection_layer_num:
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv("layer{}_attn".format(i)))
        
        # triggers for obtaining or modifying features
        self.trigger_get_qkv = False # if set True --> save attn qkv in self.attn_features
        self.trigger_modify_qkv = False # if set True --> save attn qkv by self.attn_features_modify
        
        self.modify_num = None # ignore
        self.modify_num_sa = None # ignore
        
    def get_text_condition(self, text):
        if text is None:
            uncond_input = tokenizer(
                [""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(device)
            return {'encoder_hidden_states': uncond_embeddings}
        
        text_embeddings, uncond_embeddings = get_text_embedding(text, self.text_encoder, self.tokenizer)
        text_cond = [text_embeddings, uncond_embeddings]
        denoise_kwargs = {
            'encoder_hidden_states': torch.cat(text_cond)
        }
        return denoise_kwargs
    
    def reverse_process(self, input, denoise_kwargs):
        pred_images = []
        pred_latents = []
        
        decode_kwargs = {'vae': vae}

        # Reverse diffusion process
        for t in tqdm(self.scheduler.timesteps):
            # setting t (for saving time step)
            self.cur_t = t.item()
            
            with torch.no_grad():
                noisy_residual = unet_wrapper.unet(input, t.to(input.device), **denoise_kwargs).sample
                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    noisy_residual = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    input, _ = input.chunk(2)
                
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample                # coef * P_t(e_t(x_t)) + D_t(e_t(x_t))
                pred_original_sample = scheduler.step(noisy_residual, t, input).pred_original_sample    # D_t(e_t(x_t))
                
                input = prev_noisy_sample

                # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    input = torch.cat([input] * bs)
                
                pred_latents.append(pred_original_sample)
                pred_images.append(decode_latent(pred_original_sample, **decode_kwargs))
                
        return pred_images, pred_latents
        
            
    ## Inversion (https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)
    def invert_process(self, input, denoise_kwargs):
        pred_images = []
        pred_latents = []
        
        decode_kwargs = {'vae': vae}

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)
        num_inference_steps = len(self.scheduler.timesteps)

        # For text condition on stable diffusion
        if 'encoder_hidden_states' in denoise_kwargs.keys():
            bs = denoise_kwargs['encoder_hidden_states'].shape[0]
            input = torch.cat([input] * bs)
        with torch.no_grad():
            for i in tqdm(range(0, num_inference_steps)):

                t = timesteps[i]
                
                self.cur_t = t.item()
                
                # Predict the noise residual
                noisy_residual = self.unet(input, t.to(input.device), **denoise_kwargs).sample

                noise_pred = noisy_residual

                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    noisy_residual = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    input, _ = input.chunk(2)

                current_t = max(0, t.item() - (1000//num_inference_steps)) #t
                next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]

                latents = input
                # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
                latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
                
                input = latents
                
                pred_latents.append(latents)
                pred_images.append(decode_latent(latents, **decode_kwargs))
                
        return pred_images, pred_latents
        
    # ============================ hook operations ===============================
    
    # save key value in self.original_kv[name]
    def __get_query_key_value(self, name):
        def hook(model, input, output):
            if self.trigger_get_qkv:
                _, query, key, value, _ = attention_op(model, input[0])
                self.attn_features[name][int(self.cur_t)] = (query.detach(), key.detach(), value.detach())
            
        return hook

    
    def __modify_self_attn_qkv(self, name):
        def hook(model, input, output):
            if self.trigger_modify_qkv:
                _, q_cs, k_cs, v_cs, _ = attention_op(model, input[0])
                attention_mask_ls = unet_wrapper.attention_mask
                
                q_c, k_c, v_c, q_s, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                
                # style injection
                q_hat_cs = q_c * self.style_transfer_params['gamma'] + q_cs * (1 - self.style_transfer_params['gamma'])

                _, _, _, _, modified_output = attention_op(
                    model, input[0], key=k_s, value=v_s, query=q_hat_cs, temperature=self.style_transfer_params['tau'],
                    attention_mask_ls=attention_mask_ls,
                    )
                return modified_output
        return hook

if __name__ == "__main__":
    cfg = get_args()
    sd_version = '2.1'
    json_file = "./_input/_json/zero.json"

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            t0 = time()
            if item['active'] == False:
                continue
            img_size = item["resolution"]
            frame_id = item["frame_id"]
            exp_name = item["exp_name"]
            appearance = item["appearance"]
            anchor_frame = item["anchor_frame"]
            
            content_folder = f"./_input/_data/{exp_name}/content"
            style_path = f"./_input/_data/{exp_name}/style/frame_{anchor_frame:04d}_{appearance}.png"
            anchor_path = f"{content_folder}/frame_{anchor_frame:04d}.png"
            
            # options
            ddim_steps = 20
            device = "cuda"
            dtype = torch.float16
            in_c = 4
            guidance_scale = 0.0 # no text
            
            style_text = ""
            content_text = ""

            def get_key(item, key):
                return item.get(key)
            # Load hyperparameters about Cross-image Attention
            keys = ["k", "gamma", "tau"]
            params = {key: get_key(item, key) for key in keys}
            if all(params.values()):
                params_list = iter_combine(**params)
            else:
                non_none_params = {k: v for k, v in params.items() if v is not None}
                params_list = iter_combine(**non_none_params)
  
            for style_transfer_params in params_list:
                # Init style transfer module
                vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(sd_version=sd_version, precision_t=dtype)
                scheduler.set_timesteps(ddim_steps)
                sample_size = unet.config.sample_size

                unet_wrapper = style_transfer_module(unet, vae, text_encoder, tokenizer, scheduler, style_transfer_params=style_transfer_params)
                print("style_path: ", style_path)
                style_latent, style_features, _ = image_inversion(style_path, style_text, unet_wrapper)

                for i in range(frame_id[0], frame_id[1], frame_id[2]):
                    if i == frame_id[0]:
                        dift = SDFeaturizer("./_pretrained_model/stable-diffusion-2-1-base", null_prompt='')
                    content_path = f"{content_folder}/frame_{i:04d}.png"
                    try:
                        content_image = cv2.imread(content_path)[:, :, ::-1]
                        content_image = cv2.resize(content_image, (img_size[1], img_size[0]))
                        content_latent, content_features, _ = image_inversion(content_path, content_text, unet_wrapper)
                    except:
                        print(f"cannot found frame {content_folder}/frame_{i:04d}: processing images finished~")
                        break
                    
                    cos_map_matrix, cos_map_matrix_d2, cos_map_matrix_u2 = extract_mask_topk(
                        dift=dift,
                        src_image_path = content_path,
                        trg_image_path = anchor_path,
                        img_size = img_size, # [512, 1024]
                        up_ft_index = 1,
                        top_k = style_transfer_params["top_k"],
                        )

                    cos_map_matrix = rearrange(cos_map_matrix, 'n h w a b -> n (h w) (a b)')
                    cos_map_matrix_d2 = rearrange(cos_map_matrix_d2, 'n h w a b -> n (h w) (a b)')
                    cos_map_matrix_u2 = rearrange(cos_map_matrix_u2, 'n h w a b -> n (h w) (a b)')
                        
                    unet_wrapper.attention_mask = [cos_map_matrix, cos_map_matrix_d2, cos_map_matrix_u2]

                    # Set modify features
                    for layer_name in style_features.keys():
                        unet_wrapper.attn_features_modify[layer_name] = {}
                        for t in scheduler.timesteps:
                            t = t.item()
                            unet_wrapper.attn_features_modify[layer_name][t] = (
                                content_features[layer_name][t][0], content_features[layer_name][t][1], content_features[layer_name][t][2],
                                style_features[layer_name][t][0], style_features[layer_name][t][1], style_features[layer_name][t][2]
                            ) # content as q / style as kv        
                    # =============================================
                    
                    unet_wrapper.trigger_get_qkv = False
                    unet_wrapper.trigger_modify_qkv = not cfg.without_attn_injection # modify attn feature (key value)
                    
                    # Generate style transferred image
                    denoise_kwargs = unet_wrapper.get_text_condition(content_text)
                
                    latent_cs = (content_latent - content_latent.mean(dim=(2, 3), keepdim=True)) / (content_latent.std(dim=(2, 3), keepdim=True) + 1e-4) * style_latent.std(dim=(2, 3), keepdim=True) + style_latent.mean(dim=(2, 3), keepdim=True)

                    # reverse process
                    print("Style transfer...")
                    images, latents = unet_wrapper.reverse_process(latent_cs, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
                    
                    # save image
                    images = [denormalize(input)[0] for input in images]
                    image_last = images[-1]
                    images = np.concatenate(images, axis=1)

                    folder_name = f"{style_transfer_params['gamma']}_{style_transfer_params['tau']}_k={style_transfer_params['top_k']}"
                    save_dir = f"./_result/zero_stage/{folder_name}"
                    os.makedirs(save_dir, exist_ok=True)

                    save_folder = os.path.join(save_dir, f"{item['exp_name']}_{item['appearance']}") 
                    os.makedirs(save_folder, exist_ok=True)
                    
                    file_name = f"frame_{i:04d}"
                    save_image(image_last, f"{save_folder}/{file_name}.png")
                    print(f"save to {save_folder}/{file_name}.png")

                    create_video_from_images(
                        save_folder,
                        save_folder + "/video.mp4",
                        fps=8
                    )
