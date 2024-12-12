import os
import shutil
import pandas as pd


# 输入路径设置
data_path = "/root/autodl-tmp/data"
label_path = "/root/autodl-tmp/data/label"
output_path = "/root/autodl-tmp/dataset"


# 输出路径设置
output_images = os.path.join(output_path, "images")
output_labels = os.path.join(output_path, "labels")
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# 1. 合并 train 和 test 数据集
for folder in ["train", "test"]:
    for subfolder in ["images", "labels"]:
        src_path = os.path.join(data_path, folder, subfolder)
        dst_path = os.path.join(output_path, subfolder)
        os.makedirs(dst_path, exist_ok=True)
        if os.path.exists(src_path):
            for file in os.listdir(src_path):
                shutil.copy(os.path.join(src_path, file), dst_path)

# 2. 合并 train 和 test 的 CSV 文件
train_csv_path = os.path.join(label_path, "train.csv")
test_csv_path = os.path.join(label_path, "test.csv")
csv_list = []
if os.path.exists(train_csv_path):
    csv_list.append(pd.read_csv(train_csv_path))
if os.path.exists(test_csv_path):
    csv_list.append(pd.read_csv(test_csv_path))
merged_csv = pd.concat(csv_list, ignore_index=True)

# 3. 重命名 images 和 labels 并更新 CSV 文件
file_mapping = {}
patient_case_counters = {}  # 用于记录每个病人的 case 帧编号计数

for image_file in sorted(os.listdir(output_images)):
    # 原始文件名解析
    original_name, ext = os.path.splitext(image_file)
    if original_name.startswith("case"):
        # 提取病人编号
        patient_id = int(original_name.split("_")[0].replace("case", "").lstrip("0") or "0")
        # 如果是新病人，初始化计数器
        if patient_id not in patient_case_counters:
            patient_case_counters[patient_id] = 1
        # 按规则生成新文件名
        new_filename = f"{patient_id}_1_{patient_case_counters[patient_id]}{ext}"
        patient_case_counters[patient_id] += 1
    else:
        continue  # 跳过不符合规则的文件

    # 重命名 image 和对应的 label
    old_image_path = os.path.join(output_images, image_file)
    new_image_path = os.path.join(output_images, new_filename)
    os.rename(old_image_path, new_image_path)

    old_label_path = os.path.join(output_labels, image_file)
    new_label_path = os.path.join(output_labels, new_filename)
    if os.path.exists(old_label_path):
        os.rename(old_label_path, new_label_path)

    # 更新映射
    file_mapping[image_file] = new_filename

# 更新 CSV 文件
merged_csv["filename"] = merged_csv["filename"].map(file_mapping).fillna(merged_csv["filename"])
merged_csv.to_csv(os.path.join(output_path, "labels.csv"), index=False)

print("数据合并和重命名完成！")
print(f"输出目录：{output_path}")
