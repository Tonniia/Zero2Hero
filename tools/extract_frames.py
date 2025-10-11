import cv2
import os

def extract_frames_opencv(video_path, output_folder, interval=1):
    """
    从视频中提取帧

    Args:
        video_path (str): 输入视频文件的路径
        output_folder (str): 保存帧图像的文件夹
        interval (int): 提取间隔，每interval帧保存一帧。设为1则保存每一帧。
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        # 逐帧读取
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            break

        # 按间隔保存帧
        if frame_count % interval == 0:
            # 生成文件名，用帧号命名，如 frame_000001.jpg
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"Saved: {frame_filename}")

        frame_count += 1

    # 释放资源
    cap.release()
    print(f"Extraction complete. Total frames processed: {frame_count}, Saved: {saved_count}")

# 使用示例
if __name__ == "__main__":
    video_file = "/data/vjuicefs_ai_camera_lgroup_ql/11187973/Zero2Hero/_input/_data/ironman/ironman.mp4"  # 替换为你的视频路径
    output_dir = "/data/vjuicefs_ai_camera_lgroup_ql/11187973/Zero2Hero/_input/_data/ironman/content" # 指定输出文件夹
    extract_frames_opencv(video_file, output_dir, interval=1) # 每30帧保存一帧