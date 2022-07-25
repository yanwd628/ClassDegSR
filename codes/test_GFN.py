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
import argparse
import os
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from dataset.dataset_gfn import DataTestSet
import statistics
import matplotlib.pyplot as plot
import re

parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gated", type=bool, default=True, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument('--type', type=str, default="", help='hard:class1   middle:class2   easy:class3 ')
parser.add_argument('--dataset', type=str, default="/root/yanwd/ClassDegSR/data/test_subs", help='Path of the validation dataset')
parser.add_argument("--intermediate_process", default="/root/yanwd/ClassDegSR/models/ori_0/3/GFN_epoch_55.pkl", type=str, help="Test on intermediate pkl (default: none)")
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set=[
    {'gated':False},
    {'gated':False},
    {'gated':True}
]

def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[1])
    start_epoch = "".join(re.findall(r"\d", resume)[2:])
    return int(trainingstep), int(start_epoch)

def displayFeature(feature):
    feat_permute = feature.permute(1, 0, 2, 3)
    grid = utils.make_grid(feat_permute.cpu(), nrow=16, normalize=True, padding=10)
    grid = grid.numpy().transpose((1, 2, 0))
    display_grid = grid[:, :, 0]
    plot.imshow(display_grid)

def test(test_gen, model, criterion, SR_dir):
    avg_psnr = 0
    med_time = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            LR_Blur = batch[0]
            HR = batch[2]
            img_name = batch[3][0]

            LR_Blur = LR_Blur.to(device)
            HR = HR.to(device)

            if opt.isTest == True:
                test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1
            else:
                test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()
            if opt.gated == True:
                gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1
            else:
                gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()

            start_time = time.perf_counter()#-------------------------begin to deal with an image's time
            [lr_deblur, sr] = model(LR_Blur, gated_Tensor, test_Tensor)
            #modify
            sr = torch.clamp(sr, min=0, max=1)
            lr_deblu = torch.clamp(lr_deblur, min=0, max=1)
            torch.cuda.synchronize()#wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time#---------finish an image
            med_time.append(evalation_time)

            ceshi = lr_deblu.cpu()
            ceshi1 = ceshi[0]

            resultLRDeblur = transforms.ToPILImage()(lr_deblu.cpu()[0])
            resultLRDeblur.save(join(LR_dir, img_name))
            resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
            resultSRDeblur.save(join(SR_dir, img_name))
            print("Processing {}".format(iteration))
            mse = criterion(sr, HR)
            psnr = 10 * log10(1 / mse)
            avg_psnr += psnr

        print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / iteration))
        median_time = statistics.median(med_time)
        print(median_time)

def model_test(model):
    model = model.to(device)
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    print(opt)
    test(testloader, model, criterion, SR_dir)

opt = parser.parse_args()
root_val_dir = opt.dataset# #----------Validation path]
SR_dir = "/root/yanwd/ClassDegSR/Results/ori/SR"
# SR_dir = join(root_val_dir, 'Results/ori', opt.type)  #--------------------------SR results save path
LR_dir = "/root/yanwd/ClassDegSR/Results/ori/LR"



testloader = DataLoader(DataTestSet(root_val_dir,opt.type), batch_size=1, shuffle=False, pin_memory=False)
print("===> Loading model and criterion")

if opt.intermediate_process:
    test_pkl = opt.intermediate_process
    if is_pkl(test_pkl):
        print("Testing model {}----------------------------------".format(opt.intermediate_process))
        train_step, epoch = which_trainingstep_epoch(opt.intermediate_process)
        opt.gated = test_set[train_step-1]['gated']
        model = torch.load(test_pkl, map_location=lambda storage, loc: storage)
        model_test(model)
    else:
        print("It's not a pkl file. Please give a correct pkl folder on command line for example --opt.intermediate_process /models/1/GFN_epoch_25.pkl)")
else:
    test_dir = 'models/'
    test_list = [x for x in sorted(os.listdir(test_dir)) if is_pkl(x)]
    print("Testing on the given 3-step trained model which stores in /models, and ends with pkl.")
    for i in range(len(test_list)):
        print("Testing model is {}----------------------------------".format(test_list[i]))
        model = torch.load(join(test_dir, test_list[i]), map_location=lambda storage, loc: storage)
        model_test(model)



