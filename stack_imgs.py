import os
from PIL import Image
import cv2
import numpy as np

# 数据集路径
path = [
    "/root/autodl-tmp/dataset/images",
    "/root/autodl-tmp/dataset/fuzzy",
    "/root/autodl-tmp/dataset/bilateral",
    "/root/autodl-tmp/dataset/stack",
]
os.makedirs(path[3], exist_ok=True)

# 确保 fuzzy、bilateral 和 stack 文件夹存在
os.makedirs(path[1], exist_ok=True)
os.makedirs(path[2], exist_ok=True)


# 1. 对 images 文件夹的图像进行预处理：调整大小、转灰度、保存为 jpg，并删除 png 文件
print("Preprocessing images in 'images' folder...")
for im in os.listdir(path[0]):
    file_path = os.path.join(path[0], im)
    try:
        # 打开图像
        img = Image.open(file_path)

        # 转换为灰度图（单通道）
        img = img.convert("L")

        # 调整尺寸为 (512, 512)
        img = img.resize((512, 512))

        # 保存为 .jpg 格式覆盖原文件
        save_path = os.path.splitext(file_path)[0] + ".jpg"
        img.save(save_path, "JPEG")

        print(f"Processed {im} -> {save_path}")

        # 删除原来的 .png 文件
        if im.lower().endswith(".png"):
            os.remove(file_path)
            print(f"Deleted original PNG file: {file_path}")
    except Exception as e:
        print(f"Error processing file {im}: {e}")

# 2. 生成 fuzzy 和 bilateral 图像
print("\nGenerating fuzzy and bilateral images...")
for im in os.listdir(path[0]):
    img_path = os.path.join(path[0], im)
    img = np.array(Image.open(img_path))

    # 确保图像为灰度图
    if len(img.shape) == 3:  # 如果是多通道（如 RGB 或 RGBA）
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 确保图像是 uint8 类型
    img = img.astype(np.uint8)

    # Otsu 阈值分割
    retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    # 双边滤波
    bilateral = cv2.bilateralFilter(img, d=10, sigmaColor=15, sigmaSpace=10)

    # 保存模糊增强和双边滤波图像
    Image.fromarray(dst).save(os.path.join(path[1], im))  # fuzzy 文件夹
    Image.fromarray(bilateral).save(os.path.join(path[2], im))

# 3. 堆叠 original、fuzzy 和 bilateral 图像
print("\nStacking original, fuzzy, and bilateral images...")
for im in os.listdir(path[0]):
    path1 = os.path.join(path[0], im)  # 原始图像
    path2 = os.path.join(path[1], im)  # 模糊增强图像
    path3 = os.path.join(path[2], im)  # 双边滤波图像

    # 加载图像
    img1 = np.array(Image.open(path1))
    img2 = np.array(Image.open(path2))
    img3 = np.array(Image.open(path3))

    # 确保所有图像形状一致
    if img1.shape != img2.shape or img1.shape != img3.shape:
        print(f"Shape mismatch: {im} - {img1.shape}, {img2.shape}, {img3.shape}")
        continue

    # 堆叠为三通道图像
    img = np.stack([img1, img2, img3], axis=-1)
    
    # 保存堆叠图像
    stack_output_path = '/root/autodl-tmp/dataset/stack/'  # 堆叠输出路径
    Image.fromarray(img).save(os.path.join(stack_output_path, im))

print("Processing completed!")
