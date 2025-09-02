from PIL import Image
import torch
import os
import imageio
import numpy as np

def normalize(image):
    image = image / 127.5 - 1
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
    return image

def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return (image * 255).round().astype("uint8")
      
# OPs for saving image from raw data
def save_image(image, filename):
    """
    Image should be in range (0, 255) and numpy array
    """
    image = Image.fromarray(image)
    image.save(filename)

def create_video_from_images(image_folder, output_video_file, fps=8):
    image_ls = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    image_ls.sort()
    images = [Image.open(os.path.join(image_folder, img)) for img in image_ls]
    result = [np.array(r) for r in images]
    imageio.mimsave(output_video_file, result, fps=fps)
