import os
import shutil
import random
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

import os
import fnmatch
import shutil

import torch
from torchvision import transforms
from lpips import LPIPS
# import clip

from skimage.metrics import structural_similarity as ssim

def cal_ssim(imageA, imageB):
    """
    计算两张图片的 SSIM 值。
    
    :param imageA: 第一张图片 (PIL.Image 对象)
    :param imageB: 第二张图片 (PIL.Image 对象)
    :return: SSIM 值 (范围为 0 到 1)
    """
    target_size = (512, 512)
    
    # 调整图片大小并转换为 NumPy 数组
    imageA = np.array(imageA.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)
    imageB = np.array(imageB.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)
    
    # # 检查图片尺寸是否一致
    # if imageA.shape != imageB.shape:
    #     raise ValueError("两张图片的尺寸不一致！")
    
    # 如果是 RGB 图像，分别计算每个通道的 SSIM 并取平均值
    if len(imageA.shape) == 3 and imageA.shape[2] == 3:  # 判断是否为 RGB 图像
        ssim_value = 0
        for channel in range(3):  # 遍历 R、G、B 三个通道
            ssim_value += ssim(imageA[:, :, channel], imageB[:, :, channel], data_range=255)
        ssim_value /= 3  # 取三个通道的平均值
    else:  # 如果是灰度图像
        ssim_value = ssim(imageA, imageB, data_range=255)
    
    return ssim_value

def apply_mask(image, mask, fill_color=(0, 0, 0)):
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    
    # 创建一个纯色背景
    fill_image = Image.new("RGB", image.size, fill_color)
    
    # 将 Mask 区域替换为填充颜色
    masked_image = Image.composite(image, fill_image, mask)
    
    return masked_image
    
def cal_clip_similarity(imageA, imageB, model_name="ViT-B/32"):
    # 加载 CLIP 模型和预处理函数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    # 加载并预处理图片
    image1 = preprocess(imageA).unsqueeze(0).to(device)
    image2 = preprocess(imageB).unsqueeze(0).to(device)

    # 获取图片的特征向量
    with torch.no_grad():
        image_features_1 = model.encode_image(image1)
        image_features_2 = model.encode_image(image2)

    # 归一化特征向量
    image_features_1 = image_features_1 / image_features_1.norm(dim=-1, keepdim=True)
    image_features_2 = image_features_2 / image_features_2.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    similarity = torch.matmul(image_features_1, image_features_2.T).item()

    return similarity

def cal_clip_similarity_mask(imageA, imageB, mask, model_name="ViT-B/32"):
    # 加载 CLIP 模型和预处理函数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    imageA = apply_mask(imageA, mask)
    imageB = apply_mask(imageB, mask)

    # 加载并预处理图片
    image1 = preprocess(imageA).unsqueeze(0).to(device)
    image2 = preprocess(imageB).unsqueeze(0).to(device)

    # 获取图片的特征向量
    with torch.no_grad():
        image_features_1 = model.encode_image(image1)
        image_features_2 = model.encode_image(image2)

    # 归一化特征向量
    image_features_1 = image_features_1 / image_features_1.norm(dim=-1, keepdim=True)
    image_features_2 = image_features_2 / image_features_2.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    similarity = torch.matmul(image_features_1, image_features_2.T).item()

    return similarity


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

def cal_lpips(imageA, imageB):
    target_size = (512, 512)
    image_np_1 = np.array(imageA.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)
    image_np_2 = np.array(imageB.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)

    loss_fn = LPIPS(net='alex')  # 可以选择不同的网络如 'vgg' 或 'squeeze'
    
    # 预处理图片
    img1 = load_and_preprocess_image(image_np_1)
    img2 = load_and_preprocess_image(image_np_2)

    # 确保设备一致性
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()

    # 计算LPIPS
    with torch.no_grad():
        lpips_distance = loss_fn(img1, img2)
    
    return lpips_distance.item()

def cal_psnr_mask(imageA, imageB, mask):
    # 定义目标尺寸
    target_size = (512, 512)
    # 调整大小到目标尺寸
    imageA = np.array(imageA.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)
    imageB = np.array(imageB.resize(target_size, Image.Resampling.LANCZOS)).astype(np.float64)
    mask = np.array(mask.resize(target_size, Image.Resampling.NEAREST)) > 0  # 二值化掩码

    # 确保 mask 的形状与 imageA 和 imageB 一致
    assert imageA.shape == imageB.shape == mask.shape, "Image and mask shapes do not match."

    # 仅在非掩码区域计算 MSE
    masked_diff = ((imageA - imageB) ** 2)[mask]  # 仅保留非掩码区域的差异
    if masked_diff.size == 0:  # 如果没有有效的非掩码区域
        return float('nan')  # 返回 NaN 表示无效结果

    mse_value = np.mean(masked_diff)  # 非掩码区域的均方误差

    # 如果 MSE 为 0，说明两幅图像在非掩码区域完全相同
    if mse_value == 0:
        return 100

    # 计算 PSNR
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value
    
def create_video_from_images(image_folder, output_video_file, fps=8, frame_id_ls=None):
    # image_ls = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    if frame_id_ls != None:
        image_ls = get_files_with_digits(
            image_folder,
            frame_id_ls
        )
    else:
        image_ls = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    print(len(image_ls))
    image_ls.sort()
    images = [Image.open(os.path.join(image_folder, img)) for img in image_ls]
    
    
    result = [np.array(r) for r in images]
    imageio.mimsave(output_video_file, result, fps=fps)

def load_and_preprocess_image(image_np):
    img_array = np.array(image_np, dtype=np.float32) / 255.0  # 归一化
    # 将图像数据转化为PyTorch tensor
    img_tensor = transforms.ToTensor()(img_array).unsqueeze(0)
    return img_tensor


import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(file_path):
    steps = []
    losses = []
    with open(file_path, "r") as f:
        for line in f:
            # 提取 Steps 和 Loss 的值
            if "Steps" in line and "Loss" in line:
                parts = line.split(",")
                step_part = [p.strip() for p in parts if "Steps" in p][0]
                loss_part = [p.strip() for p in parts if "Loss" in p][0]
                
                step = int(step_part.split(":")[1].strip())
                loss = float(loss_part.split(":")[1].strip())
                
                steps.append(step)
                losses.append(loss)
    return steps, losses

def ema_smooth(data, alpha=0.1):
    smoothed_data = []
    prev = data[0]  # 初始值
    for value in data:
        prev = alpha * value + (1 - alpha) * prev
        smoothed_data.append(prev)
    return smoothed_data

def plot_loss_curve(file_path, save_png_path):
    # 解析文件
    steps, losses = parse_log_file(file_path)
    
    # 使用 EMA 平滑损失值
    smoothed_losses = ema_smooth(losses, alpha=0.1)
    
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Original Loss", color="darkblue", alpha=0.7)
    plt.plot(steps, smoothed_losses, label="EMA Smoothed Loss", color="lightblue", linewidth=2)
    
    # 添加标题和标签
    plt.title("Loss vs Step Curve", fontsize=16)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # 显示图像
    plt.tight_layout()
    plt.savefig(save_png_path)
