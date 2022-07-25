
import argparse
import os
import torch
import re
from networks.ClassSR_GFN import ClassSR
from networks.loss import average_loss_3class, class_loss_3class
from dataset.dataset_ClassSR import DataSet_ClassSR
from torch.utils.data import DataLoader
from collections import OrderedDict

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--resume", default="/data/yanwd/ClassDegSR/models/ClassSR/ori/Class_GFN_epoch_38.pkl", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate, default=1e-4")
parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1, default=0.9")
parser.add_argument("--beta2", type=float, default=0.99 , help="Adam beta2, default=0.99")
parser.add_argument("--l1_w", type=float, default=1000 , help="l1_w")
parser.add_argument("--class_loss_w", type=float, default=0.5 , help="class_loss_w")
parser.add_argument("--average_loss_w", type=float, default=3 , help="average_loss_w")
parser.add_argument('--root', type=str, default="/data/yanwd/ClassDegSR/data/train_subs", help='Path of the training dataset')
parser.add_argument("--nEpochs", type=int, default=160, help="Number of epochs to train")
parser.add_argument("--step", type=int, default=20, help="Change the learning rate for every 30 epochs")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument('--gnf_s', type=str, default="/data/yanwd/ClassDegSR/models/class3/3/GFN_epoch_55.pkl", help='Path of the small GFN')
parser.add_argument('--gnf_m', type=str, default="/data/yanwd/ClassDegSR/models/class2/3/GFN_epoch_55.pkl", help='Path of the middle GFN')
parser.add_argument('--gnf_l', type=str, default="/data/yanwd/ClassDegSR/models/class1/3/GFN_epoch_55.pkl", help='Path of the large GFN')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mkdir_steptraing():
    root_folder = os.path.abspath('..')
    models_folder = os.path.join(root_folder, 'models', "ClassSR")
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print("===> ClassSR models store in models/ClassSR ")


def which_trainingstep_epoch(resume):
    start_epoch = "".join(re.findall(r"\d", resume)[0:])
    print(start_epoch)
    return int(start_epoch)

def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def checkpoint(epoch):
    model_out_path = "/data/yanwd/ClassDegSR/models/ClassSR/ori/Class_GFN_epoch_{}.pkl".format(epoch)
    torch.save(model, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))

def train(train_gen, model, criterions, optimizer, epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(train_gen, 1):
        #input, targetdeblur, targetsr
        LR_Blur = batch[0].to(device)
        LR_Deblur = batch[1].to(device)
        HR = batch[2].to(device)

        [out_res, type_res] = model(LR_Blur, is_train=True)

        loss = opt.l1_w*criterions[0](out_res, HR) + opt.class_loss_w*criterions[1](type_res) + opt.average_loss_w*criterions[2](type_res)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 200 == 0:
            print("===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(trainloader), loss.cpu()))
    print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))


print("===> Loading model and criterion")

if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        model = torch.load(opt.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(model.state_dict())
        opt.start_epoch = which_trainingstep_epoch(opt.resume)
else:
    model = ClassSR()

    model.net1 = torch.load(opt.gnf_l)
    model.net1.load_state_dict(model.net1.state_dict())

    model.net2 = torch.load(opt.gnf_m)
    model.net2.load_state_dict(model.net2.state_dict())

    model.net3 = torch.load(opt.gnf_s)
    model.net3.load_state_dict(model.net3.state_dict())

    mkdir_steptraing()
    opt.start_epoch = 1

model = model.to(device)

param_optim=[]
for name, module in model._modules.items():
    if name=="classifier":
        for p in module.parameters():
            param_optim.append(p)
    else:
        for p in module.parameters():
            p.requires_grad = False

# for name, module in model._modules.items():
#     print(name)
#     for p in module.parameters():
#         print(p.requires_grad)

# print(model)
print()

cri_pix = torch.nn.L1Loss().to(device)
class_loss = class_loss_3class().to(device)
average_loss = average_loss_3class().to(device)
criterions = [cri_pix, class_loss, average_loss]
optimizer = torch.optim.Adam(param_optim, lr=opt.lr,  betas=(opt.beta1, opt.beta2))
print()

train_set = DataSet_ClassSR(opt.root)
trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=1)

for epoch in range(opt.start_epoch, opt.nEpochs+1):
    adjust_learning_rate(epoch - 1)
    print("============epoch {}============".format(epoch))
    train(trainloader, model, criterions, optimizer, epoch)
    checkpoint(epoch)




