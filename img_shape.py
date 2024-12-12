import os
from PIL import Image
import numpy as np

# 文件夹路径
folders = {
    "images": "/root/autodl-tmp/dataset/images",
    "fuzzy": "/root/autodl-tmp/dataset/fuzzy",
    "bilateral": "/root/autodl-tmp/dataset/bilateral"
}

# 遍历每个文件夹
for folder_name, folder_path in folders.items():
    print(f"\nChecking folder: {folder_name}")
    files = os.listdir(folder_path)
    shapes = []
    
    # 获取所有图像的形状
    for im in files:
        file_path = os.path.join(folder_path, im)
        try:
            img = np.array(Image.open(file_path))  # 加载图像
            shapes.append(img.shape)
        except Exception as e:
            print(f"Error reading file {im}: {e}")  # 捕获异常，打印错误信息

    # 检查是否所有形状一致
    if len(set(shapes)) == 1:  # 使用 set 去重，如果长度为 1，说明所有形状一致
        print(f"All files in {folder_name} have the same shape: {shapes[0]}")
    else:
        print(f"Files in {folder_name} have inconsistent shapes: {set(shapes)}")

    # 如果形状一致，打印第一个文件的形状
    if shapes:
        print(f"First file shape in {folder_name}: {shapes[0]}")
