# Project Overview

## Description

This project focuses on developing a self-supervised learning framework for medical image analysis. The key functionalities include dataset preprocessing, pretraining with a self-supervised learning model, and downstream tasks such as classification. Below are the core components of the project:

### Key Scripts
- **`stack_img.py`**: Preprocesses images in the dataset by stacking the `original`, `fuzzy`, and `bilateral` processed images into a single three-channel image.
- **`get_mask_new.py`**: Generates downstream dataset structure by organizing and filtering raw images.
- **`benchmark`**: Implements DenseNet169 with a classification head as a baseline for comparison.
- **`breast`**: A self-supervised learning framework inspired by the paper [Self-supervised Learning in Pathology](https://www.arxiv.org/pdf/2408.10600.pdf). The image preprocessing in this pipeline is replaced by the custom scripts listed above.

---
# How to Run

1. **Transform Dataset Format and Generate Processed Dataset**
   - Use `stack_imgs.py` to process and generate the stacked dataset.
   - Use `get_mask_new.py` to transform the original dataset format.


3. **Pretrain the Self-Supervised Model**
   - Use the stacked dataset folder (`stack`) as input for pretraining.
   - Run the following command to train the model and obtain pretrained weights:
     `python breast/pretrain/main/train.py`

4. **Generate Downstream Task Dataset**
   - Use `get_mask_new.py` on the `stack` dataset to generate the downstream classification task dataset, which includes:
     - `train/images`
     - `test/images`
     - `train.csv`
     - `test.csv`

5. **Train and Test the Downstream Task**
   - Run the downstream training and testing process:
     `python downstream/train.py` with 5-fold validation
     or `python downstream/train_new.py` direct train and test, no 5-fold validation


## Directory Structure

```plaintext
.
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
│   ├── train_new.py
│   ├── parser_args2.py
│   ├── tools
│   │   ├── dataloader.py
│   │   ├── DenseNet.py

