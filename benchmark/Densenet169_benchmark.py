import os 
import pandas as pd 
from PIL import Image 
import torch 
import torch.nn as nn
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader 
from torchvision.models import densenet169 
from sklearn.metrics import roc_auc_score 
import argparse 

parser = argparse.ArgumentParser(description="Train or Test DenseNet169")
parser.add_argument('--mode', type=str, choices=['train','test'], required=True, help='Mode: train or test')
parser.add_argument('--weights', type=str, default='/root/autodl-tmp/benchmark/densenet169_model.pth', help='model weight path')
args = parser.parse_args()

#定义数据集
class mydataset(Dataset):
    def __init__(self, csv_file, img_dir, transform = None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]["filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["type"]
        if self.transform:
            image = self.transform(image)
        return image,label

#数据预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = mydataset(csv_file='/root/autodl-tmp/downstream_data_use/train.csv', img_dir='/root/autodl-tmp/downstream_data_use/train/images', transform = transform)
test_dataset = mydataset(csv_file='/root/autodl-tmp/downstream_data_use/test.csv', img_dir='/root/autodl-tmp/downstream_data_use/test/images', transform = transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

#模型
model = densenet169(pretrained = False)
model.classifier = nn.Linear(model.classifier.in_features, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#训练
def train_model(model, train_loader, criterion, optimizer, num_epochs=50, save_path="model169.pth", log_csv="train_loss169.csv"):
    model.train()
    loss_log = []
    for epoch in range(num_epochs):
        running_loss = 0.0 
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_log.append([epoch+1, epoch_loss])
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(),save_path)
    print(f"Model saved to {save_path}")
    loss_df = pd.DataFrame(loss_log, columns=["Epoch", "Loss"])
    loss_df.to_csv(log_csv, index=False)
    print(f"Loss log saved to {log_csv}")

#测试
def evaluate_model(model, test_loader, model_path = "/root/autodl-tmp/benchmark/resnet50_model.pth"):
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            probs = torch.softmax(outputs, dim=1)[:,1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Test AUC:{auc:.4f}")

if __name__ == "__main__":
    if args.mode == 'train':
        train_model(model, train_loader, criterion, optimizer, num_epochs=50, save_path="densenet169_model.pth", log_csv="train_loss.csv")
    if args.mode == 'test':
        evaluate_model(model, test_loader, model_path=args.weights)