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
def train(model, train_set, valid_set, book, sheet, t):
    model.linear1.apply(init_linear)
    model.linear2.apply(init_linear)
    model.linear3.apply(init_linear)

    train_loader = DataLoader(train_set,  batch_size=args.train_batch_size, shuffle=args.train_shuffle, num_workers=args.n_threads, drop_last=False)
    valid_loader = DataLoader(valid_set,  batch_size=args.eval_batch_size, shuffle=args.eval_shuffle, num_workers=args.n_threads,drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.beta, eps=1e-08, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.9993)

    sum_loss, sum_auc, sum_sp, sum_se = 0, 0, 0, 0

    # Prepare CSV file to save loss
    csv_filename = f"{args.save_model_dir}/epoch_loss_t{t}.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Epoch", "Train Loss", "Validation Loss", "AUC"])

        for epoch in range(args.epochs):
            print(f'epoch: {epoch+1}/{args.epochs}')
            losses1 = 0

            model.train()
            for img1, target1 in train_loader:
                img1, target1 = img1.to(args.device), target1.to(args.device)
                out1 = model(img1)
                loss1 = nn.CrossEntropyLoss()
                calc_loss = loss1(out1, target1.squeeze())
                losses1 += calc_loss
                opt.zero_grad()
                calc_loss.backward() 
                opt.step()
            scheduler.step()
            train_loss = losses1 / len(train_loader)
            print("training loss:{:.9f}".format(train_loss))

            losses2 = 0 
            roc_auc2 = 0
            se2 = 0
            sp2 = 0

            model.eval()

            with torch.no_grad():
                for img2, target2 in valid_loader:
                    img2, target2 = img2.to(args.device), target2.to(args.device)
                    out2 = model(img2)

                    loss2 = nn.CrossEntropyLoss()
                    cal_loss = loss2(out2, target2.squeeze())
                    losses2 += cal_loss.item()

                    out2 = out2.reshape(1, -1).squeeze()
                    target2 = F.one_hot(target2, num_classes=2).reshape(1, -1).squeeze()

                    tmp = roc_auc_score(target2.cpu().numpy(), out2.cpu().numpy())
                    roc_auc2 += tmp

            valid_loss = losses2 / len(valid_loader)
            roc_auc_avg = roc_auc2 / len(valid_loader)

            # Write epoch results to CSV
            csv_writer.writerow([epoch + 1, train_loss.item(), valid_loss, roc_auc_avg])

            
            print(f'AUC for epoch {epoch + 1}: {roc_auc_avg:.5f}')

    save_path = f"{args.save_model_dir}/model_epoch_{t}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")



def test(args):
    test_datasets = Data(args.test_labels, args.test_dir)
    test_loader = DataLoader(test_datasets, batch_size=args.test_batch_size, num_workers=args.n_threads, shuffle=args.test_shuffle)

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

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.dir_tensorboard_test)
    else:
        writer = None

    model.to(device)
    testloss, auc, t = 0, 0, 0
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for img2, target2 in test_loader:
            t += 1
            img2, target2 = img2.to(device), target2.to(device)
            out2 = model(img2)

            loss2 = nn.CrossEntropyLoss()
            cal_loss = loss2(out2, target2.squeeze())
            testloss += cal_loss.item()

            _, predicted = torch.max(out2, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target2.cpu().numpy())

            out2 = out2.reshape(1, -1)
            out2 = out2.squeeze()
            target2 = F.one_hot(target2, num_classes=2).reshape(1, -1).squeeze()

            tmp = roc_auc_score(target2.cpu().numpy(), out2.cpu().numpy())
            auc += tmp

            if args.tensorboard:
                writer.add_scalars('test loss & auc', {"testloss": testloss / t, 'auc': auc / t}, t)

        if writer:
            writer.close()

        # 计算精确率、召回率和 F1 分数
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(all_targets, all_preds, average='binary')
        recall = recall_score(all_targets, all_preds, average='binary')
        f1 = f1_score(all_targets, all_preds, average='binary')

        # 打印最终结果
        print(f"Final Test Loss: {testloss / t:.4f}")
        print(f"AUC: {auc / t:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")



if __name__ == '__main__':
    if args.mode == 'train':
        print('start to train and eval....')

        cuda_n = 'cuda:' + str(args.GPU_id)
        device = torch.device(cuda_n if torch.cuda.is_available() else 'cpu')
        epochs = args.epochs
        model = tools.__dict__[args.model]()
        model = loadpretrain(model)
        #print(model)

        train_datasets = Data(args.train_labels, args.train_dir)

        indices = args.indices
        split = args.split
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheetc = book.add_sheet('train', cell_overwrite_ok=True)
        # 假设 indices 是一个整数
        if isinstance(indices, int):
            indices = list(range(indices))  # 生成 [0, 1, ..., indices-1]

        split = len(indices) // 5  # 假设分成 5 份
        t=1
        for i in range(5):
            train_idx, valid_idx = list(set(indices) ^ set(indices[i * split:(i + 1) * split])), indices[i * split:(i + 1) * split]
            train_set = Subset(train_datasets, train_idx)
            valid_set = Subset(train_datasets, valid_idx)


            
            train(model,train_set, valid_set, book, sheetc,t)
            t += 1

    elif args.mode == 'test':
        print('start to test....')
        test(args)


    
