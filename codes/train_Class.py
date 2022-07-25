from networks.ClassSR_GFN import Classifier
from dataset.dataset_class import dataset_class
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn.functional as F
import os
import re

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=640, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--step", type=int, default=30, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="/data/yanwd/ClassDegSR/models/Class/Class_epoch_160.pkl", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--testnum", default=20, type=int, help="test for every 10 epochs")
parser.add_argument('--train_root', type=str, default="/data/yanwd/ClassDegSR/data/train_subs_class/deLR_subs", help='Path of the training dataset')
parser.add_argument('--test_root', type=str, default="/data/yanwd/ClassDegSR/data/test_subs_class/deLR_subs", help='Path of the testing dataset')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def checkpoint(epoch):
    model_out_path = "/data/yanwd/ClassDegSR/models/Class/Class_epoch_{}.pkl".format(epoch)
    torch.save(model, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))

opt = parser.parse_args()


def which_trainingstep_epoch(resume):
    start_epoch = "".join(re.findall(r"\d", resume)[0:])
    print(start_epoch)
    return int(start_epoch)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        model = torch.load(opt.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(model.state_dict())
        opt.start_epoch = which_trainingstep_epoch(opt.resume)
else:
    model = Classifier()


model.to(device)
loss_F = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
train_dataset = dataset_class(opt.train_root, True)
test_dataset = dataset_class(opt.test_root, False)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batchSize)
test_dataset = DataLoader(test_dataset, shuffle=False, batch_size=1)
trainlen = len(train_dataloader)
testlen = len(test_dataset)

lr = opt.lr
epochs = opt.nEpochs
testnum = opt.testnum
for epoch in range(opt.start_epoch, epochs+1):
    running_loss = 0
    running_correct = 0
    if epoch !=0 and epoch % opt.step == 0:
        lr *= opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print("Epoch [{}/{}] lr:{}".format(epoch, epochs, lr))
    print("=======================================================")
    currtrian = 1
    for x,y in train_dataloader:
        if currtrian % 50 == 0:
            print("{}/{}".format(currtrian,trainlen))
        currtrian += 1
        x,y = x.to(device),y.to(device)
        out = model(x)
        p = F.softmax(out, dim=1)
        flag = torch.max(p, 1)[1]

        optimizer.zero_grad()
        loss = loss_F(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += torch.sum(flag == y.data)

    print("Loss: {:.8f}  Train Accuracy: {:.4f}% ".format(
        running_loss / len(train_dataset), 100 * running_correct / len(train_dataset)))

    testing_correct = 0

    if epoch != 0 and epoch % testnum == 0 :
        currtest = 1
        with torch.no_grad():
            for x,y in test_dataset:
                if currtest % 1000 == 0:
                    print("{}/{}".format(currtest, testlen))
                currtest += 1
                x,y = x.to(device),y.to(device)
                out = model(x)
                p = F.softmax(out, dim=1)
                flag = torch.max(p, 1)[1]

                testing_correct += torch.sum(flag == y.data)
        print("Test Accuracy: {:.4f}%".format(100 * testing_correct / len(test_dataset)))
        checkpoint(epoch)





