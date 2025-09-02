import argparse
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import PILToTensor
import sys
sys.path.append("./Zero_Stage/src/models")
from dift_sd import SDFeaturizer
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import cv2
from einops import rearrange

def extract_dift(dift, input_path, prompt, args):
    img = Image.open(input_path).convert('RGB')
    if args.img_size[0] > 0:
        img = img.resize([args.img_size[1], args.img_size[0]])
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    dift_ft = dift.forward(img_tensor,
                    prompt=prompt,
                    t=args.t,
                    up_ft_index=args.up_ft_index,
                    ensemble_size=args.ensemble_size)[0]
    return img, dift_ft

def func_cos_map_matrix_topk(cos_map_matrix_ls, top_k):
    mask_ls = [
        torch.zeros_like(cos_map_matrix) for cos_map_matrix in cos_map_matrix_ls
    ]
    if top_k == -1:
        top_k_ls = [int(torch.numel(cos_map_matrix[0][0][0].view(-1))) for cos_map_matrix in cos_map_matrix_ls]
    else:
        top_k_ls = [top_k, max(int(top_k/4), 1), int(top_k*4)]
        
    for i, cos_map_matrix in enumerate(cos_map_matrix_ls):
        h, w = cos_map_matrix.shape[1], cos_map_matrix.shape[2]
        for y in list(range(h)): 
            for x in list(range(w)):
                cos_map = cos_map_matrix[0, y, x]
                values, indices = torch.topk(cos_map.view(-1), top_k_ls[i])
                row_mask = torch.zeros_like(cos_map)
                row_mask.view(-1)[indices] = 1
                mask_ls[i][:, y, x] = row_mask.cpu()
    return mask_ls


def func_cos_map_matrix_topk_ls(cos_map_matrix_ls, top_k_matrix_ls):
    mask_ls = [
        torch.zeros_like(cos_map_matrix) for cos_map_matrix in cos_map_matrix_ls
    ]
    # if top_k == -1:
    #     top_k_ls = [int(torch.numel(cos_map_matrix[0][0][0].view(-1))) for cos_map_matrix in cos_map_matrix_ls]
    # else:
    #     top_k_ls = [top_k, max(int(top_k/4), 1), int(top_k*4)]
        
    for i, cos_map_matrix in enumerate(cos_map_matrix_ls):
        h, w = cos_map_matrix.shape[1], cos_map_matrix.shape[2]
        top_k_matrix = top_k_matrix_ls[i]
        for y in list(range(h)): 
            for x in list(range(w)):
                cos_map = cos_map_matrix[0, y, x]
                top_k = top_k_matrix[y][x]
                # if top_k == -1:
                if top_k < 0:
                    top_k = int(torch.numel(cos_map_matrix[0][0][0].view(-1)))
                values, indices = torch.topk(cos_map.view(-1), top_k)
                row_mask = torch.zeros_like(cos_map)
                row_mask.view(-1)[indices] = 1
                mask_ls[i][:, y, x] = row_mask.cpu()

    return mask_ls

        
def func_extract_mask_topk(args):
    dift = args.dift
    src_image_path = args.src_image_path
    trg_image_path = args.trg_image_path
    prompt = args.prompt

    src_img, src_ft = extract_dift(dift, src_image_path, prompt, args)
    num_channel = src_ft.size(1)

    trg_img, trg_ft = extract_dift(dift, trg_image_path, prompt, args)
    trg_ft_d2 = nn.Upsample(size=(int(trg_ft.shape[2]/2), int(trg_ft.shape[3]/2)), mode='bilinear')(trg_ft)
    trg_ft_u2 = nn.Upsample(size=(int(trg_ft.shape[2]*2), int(trg_ft.shape[3]*2)), mode='bilinear')(trg_ft)


    trg_vec = F.normalize(trg_ft.view(1, num_channel, -1)) # N, C, HW
    trg_vec_d2 = F.normalize(trg_ft_d2.view(1, num_channel, -1)) # N, C, HW
    trg_vec_u2 = F.normalize(trg_ft_u2.view(1, num_channel, -1)) # N, C, HW
    
    
    cos_map_matrix = torch.zeros((1, trg_ft.shape[2], trg_ft.shape[3], trg_ft.shape[2], trg_ft.shape[3]))
    cos_map_matrix_d2 = torch.zeros((1, trg_ft_d2.shape[2], trg_ft_d2.shape[3], trg_ft_d2.shape[2], trg_ft_d2.shape[3]))
    cos_map_matrix_u2 = torch.zeros((1, trg_ft_u2.shape[2], trg_ft_u2.shape[3], trg_ft_u2.shape[2], trg_ft_u2.shape[3]))


    src_ft_d2 = nn.Upsample(size=(int(src_ft.shape[2]/2), int(src_ft.shape[3]/2)), mode='bilinear')(src_ft)
    src_ft_u2 = nn.Upsample(size=(int(src_ft.shape[2]*2), int(src_ft.shape[3]*2)), mode='bilinear')(src_ft)

    for y in list(range(src_ft.shape[2])): 
        for x in list(range(src_ft.shape[3])):
            src_vec = F.normalize(src_ft[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec, trg_vec).view(1, src_ft.shape[2], src_ft.shape[3]) 
            if args.top_k == -1:
                args.top_k = int(torch.numel(cos_map.view(-1)))
            values, indices = torch.topk(cos_map.view(-1), args.top_k)
            mask = torch.zeros_like(cos_map)
            mask.view(-1)[indices] = 1
            cos_map_matrix[:, y, x] = mask.cpu()

    for y in list(range(src_ft_d2.shape[2])): 
        for x in list(range(src_ft_d2.shape[3])):
            src_vec_d2 = F.normalize(src_ft_d2[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec_d2, trg_vec_d2).view(1, src_ft_d2.shape[2], src_ft_d2.shape[3]) # N, H, W
            if args.top_k == -1:
                args.top_k = int(torch.numel(cos_map.view(-1)))
            values, indices = torch.topk(cos_map.view(-1), max(int(args.top_k/4), 1))
            mask = torch.zeros_like(cos_map)
            mask.view(-1)[indices] = 1
            cos_map_matrix_d2[:, y, x] = mask.cpu()

    for y in list(range(src_ft_u2.shape[2])): 
        for x in list(range(src_ft_u2.shape[3])):
            src_vec_u2 = F.normalize(src_ft_u2[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec_u2, trg_vec_u2).view(1, src_ft_u2.shape[2], src_ft_u2.shape[3]) # N, H, W
            if args.top_k == -1:
                args.top_k = int(torch.numel(cos_map.view(-1)))
            values, indices = torch.topk(cos_map.view(-1), int(args.top_k*4))
            mask = torch.zeros_like(cos_map)
            mask.view(-1)[indices] = 1
            cos_map_matrix_u2[:, y, x] = mask.cpu()


    return cos_map_matrix, cos_map_matrix_d2, cos_map_matrix_u2

def func_extract_cos(args):
    dift = args.dift
    src_image_path = args.src_image_path
    trg_image_path = args.trg_image_path
    prompt = args.prompt

    src_img, src_ft = extract_dift(dift, src_image_path, prompt, args)
    num_channel = src_ft.size(1)

    trg_img, trg_ft = extract_dift(dift, trg_image_path, prompt, args)
    trg_ft_d2 = nn.Upsample(size=(int(trg_ft.shape[2]/2), int(trg_ft.shape[3]/2)), mode='bilinear')(trg_ft)
    trg_ft_u2 = nn.Upsample(size=(int(trg_ft.shape[2]*2), int(trg_ft.shape[3]*2)), mode='bilinear')(trg_ft)


    trg_vec = F.normalize(trg_ft.view(1, num_channel, -1)) # N, C, HW
    trg_vec_d2 = F.normalize(trg_ft_d2.view(1, num_channel, -1)) # N, C, HW
    trg_vec_u2 = F.normalize(trg_ft_u2.view(1, num_channel, -1)) # N, C, HW
    
    
    cos_map_matrix = torch.zeros((1, trg_ft.shape[2], trg_ft.shape[3], trg_ft.shape[2], trg_ft.shape[3]))
    cos_map_matrix_d2 = torch.zeros((1, trg_ft_d2.shape[2], trg_ft_d2.shape[3], trg_ft_d2.shape[2], trg_ft_d2.shape[3]))
    cos_map_matrix_u2 = torch.zeros((1, trg_ft_u2.shape[2], trg_ft_u2.shape[3], trg_ft_u2.shape[2], trg_ft_u2.shape[3]))


    src_ft_d2 = nn.Upsample(size=(int(src_ft.shape[2]/2), int(src_ft.shape[3]/2)), mode='bilinear')(src_ft)
    src_ft_u2 = nn.Upsample(size=(int(src_ft.shape[2]*2), int(src_ft.shape[3]*2)), mode='bilinear')(src_ft)

    for y in list(range(src_ft.shape[2])): 
        for x in list(range(src_ft.shape[3])):
            src_vec = F.normalize(src_ft[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec, trg_vec).view(1, src_ft.shape[2], src_ft.shape[3]) # N, H, W
            cos_map = cos_map/(cos_map).sum()
            cos_map_matrix[:, y, x] = cos_map.cpu()

    for y in list(range(src_ft_d2.shape[2])): 
        for x in list(range(src_ft_d2.shape[3])):
            src_vec_d2 = F.normalize(src_ft_d2[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec_d2, trg_vec_d2).view(1, src_ft_d2.shape[2], src_ft_d2.shape[3]) # N, H, W
            cos_map = cos_map/(cos_map).sum()
            cos_map_matrix_d2[:, y, x] = cos_map.cpu()

    for y in list(range(src_ft_u2.shape[2])): 
        for x in list(range(src_ft_u2.shape[3])):
            src_vec_u2 = F.normalize(src_ft_u2[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec_u2, trg_vec_u2).view(1, src_ft_u2.shape[2], src_ft_u2.shape[3]) # N, H, W
            cos_map = cos_map/(cos_map).sum()
            cos_map_matrix_u2[:, y, x] = cos_map.cpu()

    return cos_map_matrix, cos_map_matrix_d2, cos_map_matrix_u2



def get_top1_radius_mask(cos_map, r):
    height = cos_map.shape[1]
    width = cos_map.shape[2]
    values, indices = torch.topk(cos_map.view(-1), 1)
    row, col = divmod(int(indices[0]), cos_map.shape[2])
    mask = torch.zeros_like(cos_map)
    mask[0][row][col] = 1

    # �算圆的半径
    radius = r

    # �历所有可能的点
    for i in range(max(0, row - radius), min(height, row + radius + 1)):
        for j in range(max(0, col - radius), min(width, col + radius + 1)):
            # �算点到中心的距离
            distance = math.sqrt((i - row) ** 2 + (j - col) ** 2)
            # 如果点在圆内，设置掩码值为1
            if distance <= radius:
                mask[0][i][j] = 1

    return mask


def func_extract_mask_top1_r(args):
    dift = SDFeaturizer(args.model_id, null_prompt='')
    src_image_path = args.src_image_path
    trg_image_path = args.trg_image_path
    prompt = args.prompt

    src_img, src_ft = extract_dift(dift, src_image_path, prompt, args)
    num_channel = src_ft.size(1)

    trg_img, trg_ft = extract_dift(dift, trg_image_path, prompt, args)
    trg_ft_d2 = nn.Upsample(size=(int(trg_ft.shape[2]/2), int(trg_ft.shape[3]/2)), mode='bilinear')(trg_ft)
    trg_ft_u2 = nn.Upsample(size=(int(trg_ft.shape[2]*2), int(trg_ft.shape[3]*2)), mode='bilinear')(trg_ft)


    trg_vec = F.normalize(trg_ft.view(1, num_channel, -1)) # N, C, HW
    trg_vec_d2 = F.normalize(trg_ft_d2.view(1, num_channel, -1)) # N, C, HW
    trg_vec_u2 = F.normalize(trg_ft_u2.view(1, num_channel, -1)) # N, C, HW
    
    
    cos_map_matrix = torch.zeros((1, trg_ft.shape[2], trg_ft.shape[3], trg_ft.shape[2], trg_ft.shape[3]))
    cos_map_matrix_d2 = torch.zeros((1, trg_ft_d2.shape[2], trg_ft_d2.shape[3], trg_ft_d2.shape[2], trg_ft_d2.shape[3]))
    cos_map_matrix_u2 = torch.zeros((1, trg_ft_u2.shape[2], trg_ft_u2.shape[3], trg_ft_u2.shape[2], trg_ft_u2.shape[3]))


    src_ft_d2 = nn.Upsample(size=(int(src_ft.shape[2]/2), int(src_ft.shape[3]/2)), mode='bilinear')(src_ft)
    src_ft_u2 = nn.Upsample(size=(int(src_ft.shape[2]*2), int(src_ft.shape[3]*2)), mode='bilinear')(src_ft)

    for y in list(range(src_ft.shape[2])): 
        for x in list(range(src_ft.shape[3])):
            src_vec = F.normalize(src_ft[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec, trg_vec).view(1, src_ft.shape[2], src_ft.shape[3]) # N, H, W
            cos_map_matrix[:, y, x] = get_top1_radius_mask(cos_map, args.r)

    for y in list(range(src_ft_d2.shape[2])): 
        for x in list(range(src_ft_d2.shape[3])):
            src_vec_d2 = F.normalize(src_ft_d2[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec_d2, trg_vec_d2).view(1, src_ft_d2.shape[2], src_ft_d2.shape[3]) # N, H, W
            cos_map_matrix_d2[:, y, x] = get_top1_radius_mask(cos_map, int(args.r/2))

    for y in list(range(src_ft_u2.shape[2])): 
        for x in list(range(src_ft_u2.shape[3])):
            src_vec_u2 = F.normalize(src_ft_u2[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec_u2, trg_vec_u2).view(1, src_ft_u2.shape[2], src_ft_u2.shape[3]) # N, H, W
            cos_map_matrix_u2[:, y, x] = get_top1_radius_mask(cos_map, int(args.r*2))

    return cos_map_matrix, cos_map_matrix_d2, cos_map_matrix_u2



# DOING
def func_extract_tmp(args):

    dift = SDFeaturizer(args.model_id, null_prompt='')
    src_image_path = args.src_image_path
    trg_image_path = args.trg_image_path
    prompt = args.prompt

    src_img, src_ft = extract_dift(dift, src_image_path, prompt, args)
    num_channel = src_ft.size(1)

    trg_img, trg_ft = extract_dift(dift, trg_image_path, prompt, args)
    trg_ft_d2 = nn.Upsample(size=(int(trg_ft.shape[2]/2), int(trg_ft.shape[3]/2)), mode='bilinear')(trg_ft)
    trg_ft_u2 = nn.Upsample(size=(int(trg_ft.shape[2]*2), int(trg_ft.shape[3]*2)), mode='bilinear')(trg_ft)


    trg_vec = F.normalize(trg_ft.view(1, num_channel, -1)) # N, C, HW
    trg_vec_d2 = F.normalize(trg_ft_d2.view(1, num_channel, -1)) # N, C, HW
    trg_vec_u2 = F.normalize(trg_ft_u2.view(1, num_channel, -1)) # N, C, HW
    
    
    cos_map_matrix = torch.zeros((1, trg_ft.shape[2], trg_ft.shape[3], trg_ft.shape[2], trg_ft.shape[3]))
    cos_map_matrix_d2 = torch.zeros((1, trg_ft_d2.shape[2], trg_ft_d2.shape[3], trg_ft_d2.shape[2], trg_ft_d2.shape[3]))
    cos_map_matrix_u2 = torch.zeros((1, trg_ft_u2.shape[2], trg_ft_u2.shape[3], trg_ft_u2.shape[2], trg_ft_u2.shape[3]))


    src_ft_d2 = nn.Upsample(size=(int(src_ft.shape[2]/2), int(src_ft.shape[3]/2)), mode='bilinear')(src_ft)
    src_ft_u2 = nn.Upsample(size=(int(src_ft.shape[2]*2), int(src_ft.shape[3]*2)), mode='bilinear')(src_ft)
    mask_ls = []

    for y in list(range(src_ft.shape[2])): 
        for x in list(range(src_ft.shape[3])):
            src_vec = F.normalize(src_ft[0, :, y, x].view(1, num_channel))

            cos_map = torch.matmul(src_vec, trg_vec).view(1, src_ft.shape[2], src_ft.shape[3]) # N, H, W
            cos_map_normalized = cos_map/cos_map.sum() # sum=1

            values, indices = torch.sort(cos_map_normalized.view(-1), descending=True)
            cumulative_values = torch.cumsum(values, dim=0)
            first_exceed_indices = (cumulative_values > args.threshold).nonzero(as_tuple=True)[0][0]
            
            mask = torch.zeros_like(cos_map)
            mask.view(-1)[indices[:first_exceed_indices]] = 1
            cos_map_matrix[:, y, x] = mask.cpu()
            mask_ls.append(int(mask.sum()))
    

    return cos_map_matrix, None, None

def extract_inds(dift, src_image_path, trg_image_path, up_ft_index, img_size):
    args = parse()
    args.up_ft_index = up_ft_index
    args.src_image_path = src_image_path
    args.trg_image_path = trg_image_path
    prompt = ""
    args.prompt = ""
    args.top_k = 1
    args.img_size = img_size

    src_img, src_ft = extract_dift(dift, src_image_path, prompt, args)
    num_channel = src_ft.size(1)

    trg_img, trg_ft = extract_dift(dift, trg_image_path, prompt, args)
    trg_ft_d2 = nn.Upsample(size=(int(trg_ft.shape[2]/2), int(trg_ft.shape[3]/2)), mode='bilinear')(trg_ft)
    trg_ft_u2 = nn.Upsample(size=(int(trg_ft.shape[2]*2), int(trg_ft.shape[3]*2)), mode='bilinear')(trg_ft)


    trg_vec = F.normalize(trg_ft.view(1, num_channel, -1)) # N, C, HW
    trg_vec_d2 = F.normalize(trg_ft_d2.view(1, num_channel, -1)) # N, C, HW
    trg_vec_u2 = F.normalize(trg_ft_u2.view(1, num_channel, -1)) # N, C, HW
    
    
    cos_map_matrix = torch.zeros((1, trg_ft.shape[2], trg_ft.shape[3], trg_ft.shape[2], trg_ft.shape[3]))
    cos_map_matrix_d2 = torch.zeros((1, trg_ft_d2.shape[2], trg_ft_d2.shape[3], trg_ft_d2.shape[2], trg_ft_d2.shape[3]))
    cos_map_matrix_u2 = torch.zeros((1, trg_ft_u2.shape[2], trg_ft_u2.shape[3], trg_ft_u2.shape[2], trg_ft_u2.shape[3]))


    src_ft_d2 = nn.Upsample(size=(int(src_ft.shape[2]/2), int(src_ft.shape[3]/2)), mode='bilinear')(src_ft)
    src_ft_u2 = nn.Upsample(size=(int(src_ft.shape[2]*2), int(src_ft.shape[3]*2)), mode='bilinear')(src_ft)

    # tgt_coords = torch.full((src_ft.shape[2], src_ft.shape[3], 2), -1, dtype=torch.long)  # 默认值为 (-1, -1)

    # for y in list(range(src_ft.shape[2])): 
    #     for x in list(range(src_ft.shape[3])):
    #         src_vec = F.normalize(src_ft[0, :, y, x].view(1, num_channel))
    #         cos_map = torch.matmul(src_vec, trg_vec).view(1, src_ft.shape[2], src_ft.shape[3]) 
    #         _, max_idx = torch.max(cos_map.view(-1), dim=0)
    #         tgt_y, tgt_x = divmod(max_idx.item(), cos_map.shape[2])
            
    #         # 存储 tgt token 的坐标
    #         tgt_coords[y, x] = torch.tensor([tgt_y, tgt_x], dtype=torch.long)

    tgt_coords = torch.full((src_ft.shape[2]*src_ft.shape[3], 1), -1, dtype=torch.long)  

    for y in list(range(src_ft.shape[2])): 
        for x in list(range(src_ft.shape[3])):
            src_vec = F.normalize(src_ft[0, :, y, x].view(1, num_channel))
            cos_map = torch.matmul(src_vec, trg_vec).view(1, src_ft.shape[2], src_ft.shape[3]) 
            _, max_idx = torch.max(cos_map.view(-1), dim=0)
            # tgt_y, tgt_x = divmod(max_idx.item(), cos_map.shape[2])
            
            # 存储 tgt token 的坐标
            # y * width + x
            tgt_coords[y*src_ft.shape[3]+x] = torch.tensor([max_idx], dtype=torch.long)
          
    return tgt_coords



def parse():
    parser = argparse.ArgumentParser(
        description='''extract dift from input image, and save it as torch tenosr,
                    in the shape of [c, h, w].''')
    
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--model_id', default='"./_pretrained_model/stable-diffusion-2-1-base"', type=str, 
                        help='model_id of the diffusion model in huggingface')
    parser.add_argument('--t', default=261, type=int, 
                        help='time step for diffusion, choose from range [0, 1000]')
    parser.add_argument('--up_ft_index', default=1, type=int, choices=[0, 1, 2 ,3],
                        help='which upsampling block of U-Net to extract the feature map')
    parser.add_argument('--prompt', default='a photo of a cat', type=str,
                        help='prompt used in the stable diffusion')
    parser.add_argument('--ensemble_size', default=8, type=int, 
                        help='number of repeated images in each batch used to get features')
    parser.add_argument('--input_path', type=str, default=None,
                        help='path to the input image file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='path to save the output features as torch tensor')
    parser.add_argument('--k_star_ls', type=str, default='[1, -1]',
                    help='path to save the output features as torch tensor')
    args = parser.parse_args()
    return args


def extract_mask_topk(dift, src_image_path, trg_image_path, img_size, up_ft_index, draw=False, top_k=10):
    args = parse()
    args.img_size = img_size
    args.up_ft_index = up_ft_index
    args.src_image_path = src_image_path
    args.trg_image_path = trg_image_path
    args.prompt = ""
    args.draw = draw
    args.top_k = top_k
    args.dift = dift
    cos_map = func_extract_mask_topk(args)
    return cos_map

def extract_cos(dift, src_image_path, trg_image_path, img_size, up_ft_index):
    args = parse()
    args.img_size = img_size
    args.up_ft_index = up_ft_index
    args.src_image_path = src_image_path
    args.trg_image_path = trg_image_path
    args.prompt = ""
    args.dift = dift
    cos_map = func_extract_cos(args)
    return cos_map

def extract_mask_top1_r(src_image_path, trg_image_path, img_size, up_ft_index, draw=False, r=1):
    args = parse()
    args.img_size = img_size
    args.up_ft_index = up_ft_index
    args.src_image_path = src_image_path
    args.trg_image_path = trg_image_path
    args.prompt = ""
    args.draw = draw
    args.r = r
    cos_map = func_extract_mask_top1_r(args)
    return cos_map



def extract_tmp(src_image_path, trg_image_path, img_size, up_ft_index, draw, thres, coord):
    args = parse()
    args.img_size = img_size
    args.up_ft_index = up_ft_index
    args.src_image_path = src_image_path
    args.trg_image_path = trg_image_path
    args.prompt = ""
    args.draw = draw
    args.threshold = thres
    args.coord = (
        coord[0]/args.img_size[0],
        coord[1]/args.img_size[1],
    )
    cos_map = func_extract_tmp(args)
    return cos_map

def read_jsonl(file_path):
    import json
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 每一行是一个 JSON 对象
            if line.strip():  # 确保非空行
                try:
                    item = json.loads(line.strip())
                    data_list.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    return data_list
    
