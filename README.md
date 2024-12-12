# Project Overview

## Description

This project focuses on developing a self-supervised learning framework for medical image analysis. The key functionalities include dataset preprocessing, pretraining with a self-supervised learning model, and downstream tasks such as classification. Below are the core components of the project:

### Key Scripts

- **`get_mask_new.py`**: Generates a new dataset structure by organizing and filtering raw images.
- **`img_shape.py`**: Checks and validates the dimensions of the images in the dataset.
- **`stack_img.py`**: Preprocesses images in the dataset by stacking the `original`, `fuzzy`, and `bilateral` processed images into a single three-channel image.
- **`downstream_data.py`**: Generates datasets for downstream tasks such as classification or segmentation.
- **`benchmark`**: Implements ResNet50 with a classification head as a baseline for comparison.
- **`breast`**: A self-supervised learning framework inspired by the paper [Self-supervised Learning in Pathology](https://www.arxiv.org/pdf/2408.10600.pdf). The image preprocessing in this pipeline is replaced by the custom scripts listed above.

---

## Directory Structure

```plaintext
.
├── imgprocessing
│   ├── get_mask
│   ├── obtain_pretrain_datasets
│   ├── stack_imgs
├── pretrain
│   ├── main
│   │   ├── train.py
│   │   ├── parser_args.py
│   ├── utils
│   │   ├── dataloader.py
│   │   ├── DenseNet.py
│   │   ├── indice_get.py
│   │   ├── loss_functions.py
├── downstream
│   ├── train.py
│   ├── parser_args2.py
│   ├── tools
│   │   ├── dataloader.py
│   │   ├── DenseNet.py

# How to Run

1. **Transform Dataset Format and Generate Processed Dataset**
   - Use `get_mask_new.py` to transform the original dataset format.
   - Use `stack_imgs.py` to process and generate the stacked dataset.

2. **Pretrain the Self-Supervised Model**
   - Use the stacked dataset folder (`stack`) as input for pretraining.
   - Run the following command to train the model and obtain pretrained weights:
     `python breast/pretrain/main/train.py`

3. **Generate Downstream Task Dataset**
   - Use `downstream_data.py` on the `stack` dataset to generate the downstream classification task dataset, which includes:
     - `train/images`
     - `test/images`
     - `train.csv`
     - `test.csv`

4. **Train and Test the Downstream Task**
   - Run the downstream training and testing process:
     `python downstream/train.py`

