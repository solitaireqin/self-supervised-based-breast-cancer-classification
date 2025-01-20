import types
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader,Subset
from sklearn.metrics import roc_curve,auc,roc_auc_score
from tools import Densenet121,Densenet161,Densenet169,Densenet201,Densenet264
import torchvision.models as models
import matplotlib.pyplot as plt
import random
import xlwt
from parser_args2 import args
from tools import Data
import tools
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

def modify_resnets(model):
    # Modify attributs
    model.linear2 = nn.Linear(1000,128)
    model.linear3 = nn.Linear(128,16)
    model.linear4 = nn.Linear(16,2)
    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = torch.softmax(self.linear4(x),dim=-1)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def loadpretrain(m):
    m = m
    path,x = args.load_statedict,args.load_statedict_type
    if x == 't':
        model = m
        model.load_state_dict(torch.load(path, map_location=torch.device(device)),strict=False)
        for p in model.parameters():
            p.requires_grad = False
        model = layer(model)
        
    elif x == 'n':
        model = m
        model.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)
        model = layer(model)

    if x == 'i':
        model = models.resnet50(pretrained=False)
        model = modify_resnets(model)
        model.load_state_dict(torch.load(path,map_location=torch.device(device)),strict=False)

        for p in model.parameters():
            p.requires_grad = False
        model = layer1(model)

    elif x == 'no':
        model = m
        for params in model.parameters():
            params.requires_grad = True

        model.add_module('linear1',nn.Linear(1024,128))
    
    model.add_module('linear2',nn.Linear(128,16))
    model.add_module('linear3',nn.Linear(16,2))
    model.to(device)
    
    return model

def layer(model):            
    for i in range(18):#36
        if i + 18 < len(model.dense3):
            for p in model.dense3[i+18].parameters():
                p.requires_grad = False
    # for i in range(8/):#24
    #     for p in model.dense4[i+16].parameters():
    #         p.requires_grad = True
    for p in model.bn.parameters():
        p.requires_grad = True
    for p in model.linear.parameters():
        p.requires_grad = True
    
    model.add_module('linear1',nn.Linear(1024,128))
    return model

def layer1(model):
    for i in range(4, 6):
        for p in model.layer3[i].parameters():
            p.requires_grad = True
    for i in range(2, 3):
        for p in model.layer4[i].parameters():
            p.requires_grad = True
    # for i in range(4):
    #     for p in model.layer4[i+32].parameters():
    #         p.requires_grad = True
    # for i in range(10):
    #     for p in model.features.denseblock4[i+14].parameters():
    #         p.requires_grad = True
    # for p in model.features.norm5.parameters():
    #     p.requires_grad = True
    # for p in model.classifier.parameters():
    #     p.requires_grad = True

    # model.add_module('linear1',nn.Linear(1000,128))
    return model


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
def init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
        m.weight.to(args.device)
        m.bias.to(args.device)
        m.weight.requires_grad = True
        m.bias.requires_grad = True

def train(model, train_set, args):
    model.linear1.apply(init_linear)
    model.linear2.apply(init_linear)
    model.linear3.apply(init_linear)
    train_loader = DataLoader(
        train_set, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.n_threads
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.beta, eps=1e-08, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9993)
    csv_filename = f"{args.save_model_dir}/training_results.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Epoch", "Train Loss", "AUC", "Precision", "Recall", "F1"])

        # 开始训练
        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}/{args.epochs}')
            model.train()
            total_loss = 0

            all_preds = []
            all_targets = []


            for img, target in train_loader:
                img, target = img.to(args.device), target.to(args.device)
                
                output = model(img)
                loss1 = nn.CrossEntropyLoss()
                calc_loss = loss1(output, target.squeeze())
                total_loss += calc_loss.item()
                # 反向传播
                opt.zero_grad()
                calc_loss.backward()
                opt.step()

                # 收集预测和标签
                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            scheduler.step()
            train_loss = total_loss / len(train_loader)
            print("training loss:{:.9f}".format(train_loss))
            auc = roc_auc_score(all_targets, all_preds, multi_class="ovr")
            precision = precision_score(all_targets, all_preds, average="macro")
            recall = recall_score(all_targets, all_preds, average="macro")
            f1 = f1_score(all_targets, all_preds, average="macro")

            # 写入 CSV
            csv_writer.writerow([epoch + 1, train_loss, auc, precision, recall, f1])

            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, AUC: {auc:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # 保存当前 epoch 的模型
            save_path = f"{args.save_model_dir}/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model for epoch {epoch + 1} saved at {save_path}")

    print("Training completed.")

def test(args):
    test_datasets = Data(args.test_labels, args.test_dir)
    test_loader = DataLoader(
        test_datasets, 
        batch_size=args.test_batch_size, 
        num_workers=args.n_threads, 
        shuffle=args.test_shuffle
    )

    cuda_n = 'cuda:' + str(args.GPU_id)
    device = torch.device(cuda_n if torch.cuda.is_available() else 'cpu')
    model = tools.__dict__[args.model]()
    model.add_module('linear1', nn.Linear(1024, 128))
    model.add_module('linear2', nn.Linear(128, 16))
    model.add_module('linear3', nn.Linear(16, 2))

    print("=> loading checkpoint '{}'".format(args.load_statedict))
    checkpoint = torch.load(args.load_statedict, map_location=device)
    state_dict = checkpoint
    model_state_dict = model.state_dict()
    model.load_state_dict(model_state_dict)

    # 准备保存测试结果的 CSV 文件
    csv_filename = f"{args.save_model_dir}/test_results.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Test Loss", "AUC", "Precision", "Recall", "F1", "Accuracy"])
        model.to(device)
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for img, target in test_loader:
                img, target = img.to(args.device), target.to(args.device)
                output = model(img)
                loss2 = nn.CrossEntropyLoss()
                cal_loss = loss2(output, target.squeeze())
                total_loss += cal_loss.item()
                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # 计算指标
        test_loss = total_loss / len(test_loader)
        auc = roc_auc_score(all_targets, all_preds, multi_class="ovr")
        precision = precision_score(all_targets, all_preds, average="macro")
        recall = recall_score(all_targets, all_preds, average="macro")
        f1 = f1_score(all_targets, all_preds, average="macro")
        accuracy = (sum(1 for p, t in zip(all_preds, all_targets) if p == t) / len(all_targets))

        # 写入 CSV
        csv_writer.writerow([test_loss, auc, precision, recall, f1, accuracy])

        print(f"Test Loss: {test_loss:.4f}")
        print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

        print("Testing completed.")



if __name__ == '__main__':
    if args.mode == 'train':
        print('Start training...')
        cuda_n = 'cuda:' + str(args.GPU_id)
        device = torch.device(cuda_n if torch.cuda.is_available() else 'cpu')

        model = tools.__dict__[args.model]()
        model = loadpretrain(model)
        model.to(device)

        train_datasets = Data(args.train_labels, args.train_dir)

        train(model, train_datasets, args)

    elif args.mode == 'test':
        print("Starting testing...")
        test(args)
    
