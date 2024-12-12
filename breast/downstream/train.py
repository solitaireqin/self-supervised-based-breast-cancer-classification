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


class SmallClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SmallClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 从输入维度到隐藏层
            nn.ReLU(),                        # 激活函数
            nn.Linear(hidden_dim, output_dim) # 从隐藏层到输出
        )

    def forward(self, x):
        return self.classifier(x)
# 定义分类器
input_dim = 1024
hidden_dim = 128
output_dim = 2
classifier = SmallClassifier(input_dim, hidden_dim, output_dim).to(args.device)

# 定义优化器
optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)


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
def train(model, train_set,valid_set, book, sheet, t):
    model.linear1.apply(init_linear)
    model.linear2.apply(init_linear)
    model.linear3.apply(init_linear)
    
    train_loader = DataLoader(train_set,  batch_size=args.train_batch_size, shuffle=args.train_shuffle, num_workers=args.n_threads, drop_last=False)
    valid_loader = DataLoader(valid_set,  batch_size=args.eval_batch_size, shuffle=args.eval_shuffle, num_workers=args.n_threads,drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.beta, eps=1e-08, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.9993)
    
    sum_loss, sum_auc, sum_sp, sum_se = 0,0,0,0

    for epoch in range(args.epochs):
        losses1 = 0

        model.train()
        for img1,target1 in train_loader:

            img1,target1 = img1.to(args.device), target1.to(args.device)
            out1=model(img1)
            #out1 = classifier(out1)
            loss1=nn.CrossEntropyLoss()
            calc_loss = loss1(out1,target1.squeeze())
            losses1 += calc_loss
            
#            out1=out1.reshape(1,-1)
#            out1=out1.squeeze()
#            target1=F.one_hot(target1,num_classes=2)
#            target1=target1.reshape(1,-1)
#            target1=target1.squeeze()
#            
#            roc_auc1+=roc_auc_score(target1.cpu().numpy(),out1.cpu().detach().numpy())
            
            opt.zero_grad() #梯度清零
            calc_loss.backward() #计算梯度
            opt.step() #更新参数
        scheduler.step()
        print("training loss:{:.9f}".format(losses1/len(train_loader)))
    
        losses2 = 0 
        roc_auc2 = 0
        se2 = 0
        sp2 = 0

        model.eval()

        if epoch <= 99:
            with torch.no_grad():
                for img2, target2 in valid_loader:
                    img2, target2 = img2.to(args.device), target2.to(args.device)
                    out2 = model(img2)
                    #out2 = classifier(out2)
                    loss2 = nn.CrossEntropyLoss()
                    cal_loss = loss2(out2, target2.squeeze())
                    losses2 += cal_loss.item()
                    
                    out2 = out2.reshape(1, -1)
                    out2 = out2.squeeze()

                    target2 = F.one_hot(target2, num_classes=2)
                    target2 = target2.reshape(1, -1)
                    #print(out2.size(),target2.size())

                    target2 = target2.squeeze()
                    
                    tmp = roc_auc_score(target2.cpu().numpy(), out2.cpu().numpy())
                    roc_auc2 += tmp


        if epoch > 99:
            with torch.no_grad():
                for img2,target2 in valid_loader:
                    img2,target2 = img2.to(args.device),target2.to(args.device)
                    out2 = model(img2)
                    #out2 = classifier(out2)
                    loss2 = nn.CrossEntropyLoss()
                    cal_loss = loss2(out2,target2.squeeze())
                    losses2 += cal_loss.item()

                    out2=out2.reshape(1,-1)
                    out2=out2.squeeze()
                    target2=F.one_hot(target2,num_classes=2)
                    target2=target2.reshape(1,-1)
                    target2=target2.squeeze()

                    tmp=roc_auc_score(target2.cpu().numpy(),out2.cpu().numpy())
                    roc_auc2+=tmp

                    tp = (out2*target2).sum().item()
                    se = tp / (target2.sum().item()+1e-5)
                    tn = ((1-out2)*(1-target2)).sum().item()
                    sp = tn/(1-target2).sum().item()
                    se2 += se
                    sp2 += sp
                    
                losses2 /= len(valid_loader)
                sum_loss += losses2
                roc_auc2 /= len(valid_loader)
                sum_auc += roc_auc2
                se2 /= len(valid_loader)
                sp2 /= len(valid_loader)
                sum_se += se2
                sum_sp += sp2
        print(f'epoch:{epoch}')
        print('第{:d}个测试 auc:{:.5f}'.format(epoch+1, roc_auc2/len(valid_loader)))
                
    sheet.write(t,0,epoch+1)
    sheet.write(t,1,float(sum_loss)/51)
    sheet.write(t,2,sum_auc/51)
    sheet.write(t,3,float(sum_se)/51)
    sheet.write(t,4,float(sum_sp)/51)
        
    t += 1
    save_path = f"{args.save_model_dir}/model_epoch_{t}.pth"
    torch.save(model.state_dict(), save_path)
    book.save(args.save_loss_xlsx)

def test(args):
    test_datasets = Data(args.test_labels, args.test_dir)
    test_loader = DataLoader(test_datasets, batch_size=args.test_batch_size, num_workers=args.n_threads, shuffle=args.test_shuffle)

    cuda_n = 'cuda:' + str(args.GPU_id)
    device = torch.device(cuda_n if torch.cuda.is_available() else 'cpu')
    model = tools.__dict__[args.model]()
        # 添加分类层 (与训练时一致)
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
    testloss,auc,t = 0,0,0
    model.eval()
    correct = 0  # 记录预测正确的样本数
    total = 0  
    with torch.no_grad():
        for img2, target2 in test_loader:
            t+=1
            img2, target2 = img2.to(args.device), target2.to(args.device)
            out2 = model(img2)
            loss2 = nn.CrossEntropyLoss()
            cal_loss = loss2(out2, target2.squeeze())
            testloss += cal_loss.item()

            out2 = out2.reshape(1, -1)
            out2 = out2.squeeze()
            target2 = F.one_hot(target2, num_classes=2)
            target2 = target2.reshape(1, -1)
            target2 = target2.squeeze()


            tmp = roc_auc_score(target2.cpu().numpy(), out2.cpu().numpy())
            auc += tmp

            if args.tensorboard:
                writer.add_scalars('test loss & auc', {"testloss":testloss/t,'auc':auc/t}, t)
        writer.close()


        # 打印最终结果
        print(f"Final Test Loss: {testloss / t:.4f}")
        print(f"AUC: {auc / t:.4f}")


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

        for i in range(5):
            train_idx, valid_idx = list(set(indices) ^ set(indices[i * split:(i + 1) * split])), indices[i * split:(i + 1) * split]
            train_set = Subset(train_datasets, train_idx)
            valid_set = Subset(train_datasets, valid_idx)

            t=0
            train(model,train_set, valid_set, book, sheetc,t)

    elif args.mode == 'test':
        print('start to test....')
        test(args)


    
