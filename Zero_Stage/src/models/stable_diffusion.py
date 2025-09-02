
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline, DiffusionPipeline

from diffusers import LMSDiscreteScheduler, DDIMScheduler


# From "https://huggingface.co/blog/stable_diffusion"
def load_stable_diffusion(sd_version='2.1', precision_t=torch.float32, device="cuda"):
    if sd_version == '2.1':
        model_key = "./_pretrained_model/stable-diffusion-2-1-base"
    elif sd_version == '2.0':
        pass
    elif sd_version == '1.5':
        pass
    elif sd_version == 'xl':
        pass
        
    # Create model
    if sd_version == "xl":
        pipe = DiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t, use_safetensors=True, variant="fp16")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)
    
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    # import xformer
    # unet.enable_xformers_memory_efficient_attention()
    
    del pipe
    
    # Use DDIM scheduler
    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)
    
    return vae, tokenizer, text_encoder, unet, scheduler

def decode_latent(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    return image

def encode_latent(images, vae):
    # encode the image with vae
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
    latents = 0.18215 * latents
    return latents

def get_text_embedding(text, text_encoder, tokenizer, device="cuda"):
    # TODO currently, hard-coding for stable diffusion
    with torch.no_grad():

        prompt = [text]
        batch_size = len(prompt)
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(device)
        max_length = text_input.input_ids.shape[-1]
        # print(max_length, text_input.input_ids)
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(device)
    
    return text_embeddings, uncond_embeddings



def get_unet_layers(unet):
    
    layer_num = [i for i in range(12)]
    resnet_layers = []
    attn_layers = []
    
    for idx, ln in enumerate(layer_num):
        up_block_idx = idx // 3
        layer_idx = idx % 3
        
        resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx])
        if up_block_idx > 0:
            attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
        else:
            attn_layers.append(None)
        
    return resnet_layers, attn_layers

def get_unet_layers_tt(unet):
    resnet_layers = []
    attn_layers = []
    temporal_layers = []

    # down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    # down_resnet_dict = {0: [0, 1], 1: [0], 2: 1, 3: [0, 1]}
    # down_temporal_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3:[0, 1]}

    up_spatial_dict =      {               1: [0, 1, 2],   2: [0, 1, 2],   3: [0, 1, 2]}
    up_resnet_dict =    {0: [0, 1, 2],  1: [0, 1, 2],   2: [0, 1, 2],   3: [0, 1, 2]}
    up_temporal_dict =  {0: [0, 1, 2],  1: [0, 1, 2],   2: [0, 1, 2],   3: [0, 1, 2]}
    

    # for down_block_idx in down_res_dict.keys():
    #     for layer_idx in down_res_dict[down_block_idx]:
    #         attn_layers.append(getattr(unet, 'down_blocks')[down_block_idx].attentions[layer_idx])
    for up_block_idx in up_spatial_dict.keys():
        for layer_idx in up_spatial_dict[up_block_idx]:
            attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
    
    for up_block_idx in up_resnet_dict.keys():
        for layer_idx in up_resnet_dict[up_block_idx]:
            resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx]) 
    
    for up_block_idx in up_temporal_dict.keys():
        for layer_idx in up_temporal_dict[up_block_idx]:
            temporal_layers.append(getattr(unet, 'up_blocks')[up_block_idx].motion_modules[layer_idx]) 

    return resnet_layers, attn_layers, temporal_layers
      

def get_unet_layers_tt_downmidup(unet):
    resnet_layers = []
    attn_layers = []
    temporal_layers = []

    down_spatial_dict =     {0: [0, 1],     1: [0, 1],  2: [0, 1]               }
    down_res_dict   =       {0: [0, 1],     1: [0, 1],  2: [0, 1],  3: [0, 1]    }
    down_temporal_dict =    {0: [0, 1],     1: [0, 1],  2: [0, 1],  3: [0, 1]    }

    up_spatial_dict =   {               1: [0, 1, 2],   2: [0, 1, 2],   3: [0, 1, 2]}
    up_resnet_dict =    {0: [0, 1, 2],  1: [0, 1, 2],   2: [0, 1, 2],   3: [0, 1, 2]}
    up_temporal_dict =  {0: [0, 1, 2],  1: [0, 1, 2],   2: [0, 1, 2],   3: [0, 1, 2]}

    mid_attn_dict =         {0: [0]}

    # for down_block_idx in down_res_dict.keys():
    #     for layer_idx in down_res_dict[down_block_idx]:
    #         attn_layers.append(getattr(unet, 'down_blocks')[down_block_idx].attentions[layer_idx])
    for down_block_idx in down_spatial_dict.keys():
        for layer_idx in down_spatial_dict[down_block_idx]:
            attn_layers.append(getattr(unet, 'down_blocks')[down_block_idx].attentions[layer_idx])
    for down_block_idx in down_temporal_dict.keys():
        for layer_idx in down_temporal_dict[down_block_idx]:
            temporal_layers.append(getattr(unet, 'down_blocks')[down_block_idx].motion_modules[layer_idx])

    attn_layers.append(getattr(unet, 'mid_block').attentions[0])
    temporal_layers.append(getattr(unet, 'mid_block').motion_modules[0])

    for up_block_idx in up_spatial_dict.keys():
        for layer_idx in up_spatial_dict[up_block_idx]:
            attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
    
    for up_block_idx in up_resnet_dict.keys():
        for layer_idx in up_resnet_dict[up_block_idx]:
            resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx]) 
    
    for up_block_idx in up_temporal_dict.keys():
        for layer_idx in up_temporal_dict[up_block_idx]:
            temporal_layers.append(getattr(unet, 'up_blocks')[up_block_idx].motion_modules[layer_idx]) 

    return resnet_layers, attn_layers, temporal_layers

def get_unet_layers_tt_old(unet):
    resnet_layers = []
    attn_layers = []
    temporal_layers = []

    layer_num = [i for i in range(7)] # 0-6
    layer_num = range(100)
    for idx, ln in enumerate(layer_num):
        # for each attentions, have 2 <Transformer2DModel>
        down_block_idx = idx // 2 # down_block_idx = 3 -> DownBlockMotion -> have no .attentions [Transformer2DModel]
        layer_idx = idx % 2
        
        if down_block_idx < 3: 
            attn_layers.append(getattr(unet, 'down_blocks')[down_block_idx].attentions[layer_idx])
                
        else:
            # innest layer have no attentions
            # attn_layers.append(None)
            break

    layer_num = [i for i in range(12)] # 0-11
    layer_num = range(100)

    for idx, ln in enumerate(layer_num):
        # for each attentions, have 3 <Transformer2DModel>
        up_block_idx = idx // 3 # up_block_idx = 0 -> UpBlockMotion -> have no .attentions [Transformer2DModel]
        layer_idx = idx % 3
        
        resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx]) 
        
        temporal_layers.append(getattr(unet, 'up_blocks')[up_block_idx].motion_modules[layer_idx]) # TransformerTemporalModel

        if up_block_idx > 0:
            attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
        else:
            # innest layer have no .attentions [Transformer2DModel]
            # attn_layers.append(None)
            break

    attn_layers.append(getattr(unet, 'mid_block').attentions[0]) # mid_blocks attentions

    return resnet_layers, attn_layers, temporal_layers
              
        

# Diffusers attention code for getting query, key, value and attention map
def attention_op(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0):
    residual = hidden_states
    
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if query is None:
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if key is None:
        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
    if value is None:
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

    
    if key.shape[0] != query.shape[0]:
        key, value = key[:query.shape[0]], value[:query.shape[0]]

    # apply temperature scaling
    query = query * temperature # same as applying it on qk matrix

    if attention_probs is None:
        attention_probs, attention_scores = get_attention_scores(attn, query, key, attention_mask)

    batch_heads, img_len, txt_len = attention_probs.shape
    
    # h = w = int(img_len ** 0.5)
    # attention_probs_return = attention_probs.reshape(batch_heads // attn.heads, attn.heads, h, w, txt_len)
    
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    
    return attention_probs, query, key, value, hidden_states

def get_attention_scores(
        attn, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            # modified: should add a very large value to where pay attention
            # beta = 1
            beta = 100
        # beta * baddbmm_input + (QK) / scale=sqrt(D)
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        # del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs, attention_scores


