import argparse
import os
import torch
import re
from networks.ClassSR_GFN_mine import ClassSR
from dataset.dataset_class import dataset_class
from torch.utils.data import DataLoader
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="PyTorch Test")
parser.add_argument('--model', type=str, default="/root/yanwd/ClassDegSR/models/ClassSR/mine/Class_GFN_epoch_100.pkl", help='Path of the model')
parser.add_argument('--root', type=str, default="/root/yanwd/ClassDegSR/data/test_subs_class/deLR_subs", help='Path of the testing dataset')
parser.add_argument('--part', type=bool, default=True, help='True：train with DegSR  False:train independently')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.model:
    if os.path.isfile(opt.model):
        print("Testing classifier from model {}".format(opt.model))
        if opt.part:
            model = torch.load(opt.model, map_location=lambda storage, loc: storage)
            model.load_state_dict(model.state_dict())
            model = model.classifier
        else:
            model = torch.load(opt.model, map_location=lambda storage, loc: storage)
            model.load_state_dict(model.state_dict())
else:
    print("opt.model error!!")

# print(model)
model.to(device)
test_dataset = dataset_class(opt.root, False)
test_dataset = DataLoader(test_dataset, shuffle=False, batch_size=1)
testlen = len(test_dataset)



testing_correct=0
index = 0
type0=0
type1=0
type2=0
with torch.no_grad():
    for x, y in test_dataset:
        if index%2000==0:
            print("{}/{}".format(index,testlen))
        index += 1
        x, y = x.to(device), y.to(device)
        out = model(x)
        p = F.softmax(out, dim=1)
        flag = torch.max(p, 1)[1]
        val = flag.data
        if val == 0:
            type0 += 1
        elif val == 1:
            type1 += 1
        else:
            type2 +=1
        testing_correct += torch.sum(flag == y.data)
print("Test Accuracy: {:.4f}%".format(100 * testing_correct / len(test_dataset)))
print(type0)
print(type1)
print(type2)
