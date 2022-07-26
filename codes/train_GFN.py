# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

from __future__ import print_function
import torch.optim as optim
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from dataset.dataset_gfn import DataSet
from networks.GFN import Net
import random
import re

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=24, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--scale", default=4, type=int, help="Scale factor, Default: 4")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--gated", type=bool, default=False, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument('--root', type=str, default="/data/yanwd/ClassDegSR/data/train_subs", help='Path of the training dataset')
parser.add_argument('--type', type=str, default="", help='hard:class1   middle:class2   easy:class3  ori:none')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_settings=[
    {'nEpochs': 25, 'lr': 1e-4, 'step':  7, 'lr_decay': 0.5, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 60, 'lr': 1e-4, 'step': 30, 'lr_decay': 0.1, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 55, 'lr': 5e-5, 'step': 25, 'lr_decay': 0.1, 'lambda_db':   0, 'gated': True}
]

def mkdir_steptraing(type):
    root_folder = os.path.abspath('..')
    if type == "":
        models_folder = join(root_folder, 'models', "ori")
    else:
        models_folder = join(root_folder, 'models', type)
    step1_folder, step2_folder, step3_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(step3_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        print("===> Step training models store in models/1 & /2 & /3.")


def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[1])
    start_epoch = "".join(re.findall(r"\d", resume)[2:])
    print(trainingstep,start_epoch)
    return int(trainingstep), int(start_epoch)

def adjust_learning_rate(epoch):
        lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def checkpoint(type, step, epoch):
    if type =="":
        model_out_path = "/data/yanwd/ClassDegSR/models/ori/{}/GFN_epoch_{}.pkl".format(step, epoch)
    else:
        model_out_path = "/data/yanwd/ClassDegSR/models/{}/{}/GFN_epoch_{}.pkl".format(type, step, epoch)
    torch.save(model, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))

def train(train_gen, model, criterion, optimizer, epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(train_gen, 1):
        #input, targetdeblur, targetsr
        LR_Blur = batch[0]
        LR_Deblur = batch[1]
        HR = batch[2]
        LR_Blur = LR_Blur.to(device)
        LR_Deblur = LR_Deblur.to(device)
        HR = HR.to(device)

        if opt.isTest == True:
            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()+1
        else:
            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()
        if opt.gated == True:
            gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()+1
        else:
            gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()

        [lr_deblur, sr] = model(LR_Blur, gated_Tensor, test_Tensor)

        loss1 = criterion(lr_deblur, LR_Deblur)
        loss2 = criterion(sr, HR)
        mse = loss2 + opt.lambda_db * loss1
        epoch_loss += mse
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        if iteration % 200 == 0:
            print("===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(trainloader), mse.cpu()))
    print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))

opt = parser.parse_args()
opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

type = opt.type
if type == "class1":   #hard
    nf = 64
    ng = 3
elif type == "class2": #middle
    nf = 52
    ng = 2
elif type == "class3":  #easy
    nf = 36
    ng = 1
else:                   #ori
    nf = 64
    ng = 3

train_dir = opt.root
print("===> Loading model and criterion")

if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        model = torch.load(opt.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(model.state_dict())
        opt.start_training_step, opt.start_epoch = which_trainingstep_epoch(opt.resume)
else:
    model = Net(nf, ng)
    mkdir_steptraing(type)

model = model.to(device)
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
print()

train_set = DataSet(train_dir, type)
trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=1)
print("Training folder is {}-------------------------------".format(train_dir))
for i in range(opt.start_training_step, 4):
    opt.nEpochs   = training_settings[i-1]['nEpochs']
    opt.lr        = training_settings[i-1]['lr']
    opt.step      = training_settings[i-1]['step']
    opt.lr_decay  = training_settings[i-1]['lr_decay']
    opt.lambda_db = training_settings[i-1]['lambda_db']
    opt.gated     = training_settings[i-1]['gated']
    print(opt)
    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        adjust_learning_rate(epoch-1)
        print("============Step {} epoch {}============".format(i, epoch))
        train(trainloader, model, criterion, optimizer, epoch)
        checkpoint(type, i, epoch)


