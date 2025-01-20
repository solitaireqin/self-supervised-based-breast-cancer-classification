import os
from PIL import Image
import numpy as np
import cv2

# 数据集路径
path1 = [
    "/root/autodl-tmp/data_downstream_original/train/images",      # 原始图像
    "/root/autodl-tmp/data_downstream_original/train/fuzzy",      # 模糊增强图像
    "/root/autodl-tmp/data_downstream_original/train/bilateral",  # 双边滤波图像
    "/root/autodl-tmp/data_downstream_original/train/stack",      # 堆叠图像
]
path2 = [
    "/root/autodl-tmp/data_downstream_original/test/images",      # 原始图像
    "/root/autodl-tmp/data_downstream_original/test/fuzzy",      # 模糊增强图像
    "/root/autodl-tmp/data_downstream_original/test/bilateral",  # 双边滤波图像
    "/root/autodl-tmp/data_downstream_original/test/stack",      # 堆叠图像
]
for path in [path1,path2]:
    # 确保目标文件夹存在
    os.makedirs(path[1], exist_ok=True)  # 创建 fuzzy 文件夹
    os.makedirs(path[2], exist_ok=True)  # 创建 bilateral 文件夹
    os.makedirs(path[3], exist_ok=True)  # 创建 stack 文件夹

    # 获取所有原始图像文件名
    files = os.listdir(path[0])

    # 生成 fuzzy 和 bilateral 图像
    print("\nGenerating fuzzy and bilateral images...")
    for im in files:
        img_path = os.path.join(path[0], im)  # 原始图像路径

        # 加载图像
        img = np.array(Image.open(img_path).convert("L"))  # 转为灰度图像

        # 生成 fuzzy 图像（大津阈值分割）
        retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        # 生成 bilateral 图像（双边滤波）
        bilateral = cv2.bilateralFilter(img, d=10, sigmaColor=15, sigmaSpace=10)

        # 保存 fuzzy 图像
        fuzzy_output_path = os.path.join(path[1], im)
        Image.fromarray(dst).save(fuzzy_output_path)

        # 保存 bilateral 图像
        bilateral_output_path = os.path.join(path[2], im)
        Image.fromarray(bilateral).save(bilateral_output_path)

    # 堆叠原始图像、fuzzy 和 bilateral 图像
    print("\nStacking original, fuzzy, and bilateral images...")
    for im in files:
        # 路径
        original_path = os.path.join(path[0], im)
        fuzzy_path = os.path.join(path[1], im)
        bilateral_path = os.path.join(path[2], im)

        # 加载图像
        img1 = np.array(Image.open(original_path).convert("L"))  # 原始图像
        img2 = np.array(Image.open(fuzzy_path))                 # fuzzy 图像
        img3 = np.array(Image.open(bilateral_path))             # bilateral 图像

        # 确保所有图像形状一致
        if img1.shape != img2.shape or img1.shape != img3.shape:
            print(f"Shape mismatch: {im} - {img1.shape}, {img2.shape}, {img3.shape}")
            continue

        # 堆叠图像为 3 通道
        stacked_img = np.stack([img1, img2, img3], axis=-1)

        # 保存堆叠图像
        stacked_output_path = os.path.join(path[3], im)
        Image.fromarray(stacked_img).save(stacked_output_path)

    print("Processing completed!")
