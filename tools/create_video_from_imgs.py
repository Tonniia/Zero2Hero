import cv2
import os

# 设置图片文件夹路径
folder_path = '/data/vjuicefs_ai_camera_lgroup_ql/11187973/Zero2Hero/_input/_data/car_turn/content'
# 设置输出视频文件名
output_file = '/data/vjuicefs_ai_camera_lgroup_ql/11187973/Zero2Hero/_input/_data/car_turn/video.mp4'
# 设置帧率
fps = 24
# 获取图片列表
images = [img for img in sorted(os.listdir(folder_path)) if img.endswith('.png')]
# 获取图片的宽度和高度
frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# 将图片写入视频
for image in images:
    video.write(cv2.imread(os.path.join(folder_path, image)))

# 释放资源
video.release()