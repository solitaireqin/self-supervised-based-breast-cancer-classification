import os
from pretrain.main import args
import torch.nn as nn
import torch
from pretrain.utils import LoadDatasets
from torch.utils.data import DataLoader
import xlrd
from xlutils.copy import copy
from pretrain.utils import HardTripletloss,get_indices
from pretrain import utils
torch.cuda.empty_cache()  # 清空显存缓存
'''
    obtain indices
'''
indices = get_indices()


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def train(model):
    train_datasets = LoadDatasets(indices)
    # 打印第一个样本的数据和形状
    sample = train_datasets[0]
    train_loader = DataLoader(train_datasets, batch_size=args.train_batch_size, num_workers=args.n_threads, shuffle=args.train_shuffle)

    contrastive_loss = HardTripletloss(args)

    cuda_n = 'cuda:' + str(args.GPU_id)
    device = torch.device(cuda_n if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    model = utils.__dict__[args.model]()

    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=9e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader), eta_min=1e-5)
    iter = 0
    start_epoch = 0
    verbose = False
    trainloss = 0

    if args.load:
        print("=> loading checkpoint '{}'".format(args.load_statedict))
        checkpoint = torch.load(args.load_statedict, map_location=device)
        model.load_state_dict(checkpoint['model'])
        opt = checkpoint['opt']
        scheduler = checkpoint['scheduler']
        train_loader = checkpoint['train_loader']
        iter = checkpoint['iter']
        start_epoch = checkpoint['start_epoch']

        verbose = True
        trainloss = checkpoint['trainloss']

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.dir_tensorboard_train)
    else:
        writer = None

    model.to(device)

    for epoch in range(start_epoch, epochs):
        if args.load and verbose:
            iter = iter
            verbose = False
        else:
            iter = 0

        model.train()
        for batch_id, data in enumerate(train_loader, start=1):
            iter += 1
            # 初始化累积损失
            batch_loss = 0
            img_batch = data[0]  # 形状：[16, 105, 1, 224, 224]
            img_label = data[1].to(device)

            # 遍历 batch 中的每个样本（16 个样本）
            for batch_idx in range(img_batch.size(0)):  # 遍历 16
                img = img_batch[batch_idx].to(device)  # 取一个样本，形状：[105, 1, 224, 224]

                # 模型前向传播
                fs = model(img)  # 输入整个帧序列

                # 计算损失
                loss = contrastive_loss(fs)
                batch_loss += loss.item()

                # 反向传播
                opt.zero_grad()
                loss.backward()
                opt.step()


            # 累计总损失
            trainloss += batch_loss
            print(f"Epoch [{epoch + 1}/{epochs}] | Batch [{batch_id}/{len(train_loader)}] | Batch Loss: {batch_loss:.4f}")
        scheduler.step()

        if epoch % 1 == 0:
            state = {'model': model.state_dict(), 'opt': opt.state_dict(), 'scheduler': scheduler,
                     'train_loader': train_loader, 'start_epoch': start_epoch, 'iter': iter, 'trainloss': trainloss}

            torch.save(state, args.save_models_dir + 'train' + '_' + str(epoch + 1) + '.pth')

        if args.tensorboard:
            writer.add_scalar('train loss by epoch',trainloss / iter, epoch + 1)
        # 打印每个 epoch 的平均损失
        print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {trainloss / iter:.4f}")
        if epoch % 1 == 0:
            trainloss = 0


if __name__ == '__main__':
    print('start to train...')
    train(args)



