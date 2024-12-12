import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_path = "/root/autodl-tmp/dataset"  # Replace with your dataset path
stack_path = os.path.join(dataset_path, "stack")
csv_file_path = os.path.join(dataset_path, "labels.csv")
output_path = "/root/autodl-tmp/downstream_data"  # Replace with your output path

# Create output directories
train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")
os.makedirs(os.path.join(train_path, "images"), exist_ok=True)
os.makedirs(os.path.join(test_path, "images"), exist_ok=True)

# Read and process CSV file
labels_df = pd.read_csv(csv_file_path)

# Update filenames in the CSV to have a .jpg extension
labels_df['filename'] = labels_df['filename'].str.replace('.png', '.jpg')

# Split data into train and test
train_df, test_df = train_test_split(labels_df, test_size=0.3, random_state=42, stratify=labels_df['type'])

# Save train and test CSVs
train_csv_path = os.path.join(output_path, "train.csv")
test_csv_path = os.path.join(output_path, "test.csv")
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

# Function to copy files
def copy_files(file_list, src_folder, dest_folder):
    for filename in file_list:
        src = os.path.join(src_folder, filename)
        dest = os.path.join(dest_folder, filename)
        if os.path.exists(src):
            shutil.copy(src, dest)

# Copy files based on the CSV splits
copy_files(train_df['filename'].tolist(), stack_path, os.path.join(train_path, "images"))
copy_files(test_df['filename'].tolist(), stack_path, os.path.join(test_path, "images"))

# Final message
print(f"Data split and copied successfully! Train files: {len(train_df)}, Test files: {len(test_df)}")
