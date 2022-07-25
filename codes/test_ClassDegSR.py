import os
import argparse
import torch
import numpy as np
from networks.ClassSR_GFN_mine import ClassSR
from dataset.dataset_ClassDegSR_test import DataSet_ClassDegSR_test
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from dataset.calculate_PSNR_SSIM import calculate_psnr,calculate_ssim


parser = argparse.ArgumentParser(description="PyTorch Test")
parser.add_argument('--model', type=str, default="/root/yanwd/ClassDegSR/models/ClassSR/mine/Class_GFN_epoch_100.pkl", help='Path of the model')
parser.add_argument('--gnf_s', type=str, default="/root/yanwd/ClassDegSR/models/class3/3/GFN_epoch_55.pkl", help='Path of the small GFN')
parser.add_argument('--gnf_m', type=str, default="/root/yanwd/ClassDegSR/models/class2/3/GFN_epoch_55.pkl", help='Path of the middle GFN')
parser.add_argument('--gnf_l', type=str, default="/root/yanwd/ClassDegSR/models/class1/3/GFN_epoch_55.pkl", help='Path of the large GFN')
parser.add_argument('--classifier', type=str, default="/root/yanwd/ClassDegSR/models/Class/Class_epoch_400.pkl", help='Path of the classifier')
parser.add_argument('--root', type=str, default="/root/yanwd/ClassDegSR/data/Test_4x", help='Path of the testing dataset')

parser.add_argument('--save_SR', type=str, default="/root/yanwd/ClassDegSR/Results/joint_loss_100/SR", help='Path of saving SR')
parser.add_argument('--save_deLR', type=str, default="/root/yanwd/ClassDegSR/Results/joint_loss_100/deLR", help='Path of saving deLR')
parser.add_argument('--joint', type=bool, default=True, help='True：train with DegSR  False:train independently')


parser.add_argument('--crop_size', type=int, default=64, help='crop size')
parser.add_argument('--step', type=int, default=32, help='step')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

patch_size=opt.crop_size
step=opt.step
root = opt.root
SR_dir = opt.save_SR
deLR_dir = opt.save_deLR
testloader = DataLoader(DataSet_ClassDegSR_test(root), batch_size=1, shuffle=False, pin_memory=False)

if opt.joint:
    if opt.model:
        if os.path.isfile(opt.model):
            print("Testing from model {}".format(opt.model))
            model = torch.load(opt.model, map_location=lambda storage, loc: storage)
            model.load_state_dict(model.state_dict())
        else:
            print("opt.model error!!")
else:
    if opt.gnf_s and  opt.gnf_m and opt.gnf_l and opt.classifier:
        print("Testing from model \n{}\n{}\n{}\n{}\n".format(opt.gnf_s,opt.gnf_m,opt.gnf_l,opt.classifier))
        model = ClassSR()

        model.net1 = torch.load(opt.gnf_l)
        model.net1.load_state_dict(model.net1.state_dict())

        model.net2 = torch.load(opt.gnf_m)
        model.net2.load_state_dict(model.net2.state_dict())

        model.net3 = torch.load(opt.gnf_s)
        model.net3.load_state_dict(model.net3.state_dict())

        model.classifier = torch.load(opt.classifier)
        model.classifier.load_state_dict(model.classifier.state_dict())
    else:
        print("opt.gnf_s/m/l  or classifier error!!")

# print(model)
model.to(device)

def crop_cpu(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        c, h, w  = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[:, x:x + crop_sz, y:y + crop_sz]
            lr_list.append(crop_img)
    h = x + crop_sz
    w = y + crop_sz
    return lr_list, num_h, num_w, h, w


def combine(sr_list, num_h, num_w, h, w, patch_size, step, scale):
    index = 0
    sr_img = np.zeros((3, h * scale, w * scale), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, i * step * scale:i * step * scale + patch_size * scale, j * step * scale:j * step * scale + patch_size * scale] += sr_list[index]
            index += 1
    sr_img = sr_img.astype('float32')

    for j in range(1, num_w):
        sr_img[:, :, j * step * scale:j * step * scale + (patch_size - step) * scale] /= 2

    for i in range(1, num_h):
        sr_img[:, i * step * scale:i * step * scale + (patch_size - step) * scale, :] /= 2
    return sr_img


model.eval()
with torch.no_grad():
    index = 1
    sr_psnr = 0.0
    sr_ssim = 0.0
    delr_psnr = 0.0
    delr_ssim = 0.0

    for iteration, batch in enumerate(testloader, 1):
        LR_Blur = batch[0].squeeze()
        LR = batch[1].squeeze()
        HR = batch[2].squeeze()
        img_name = batch[3][0]
        LR_Blur = LR_Blur.to(device)

        lr_list, num_h, num_w, h, w = crop_cpu(LR_Blur, patch_size, step)
        gthr_list = crop_cpu(HR, patch_size*4, step*4)[0]
        gtlr_list = crop_cpu(LR, patch_size, step)[0]

        sr_list = []
        delr_list = []

        for LR_img in lr_list:
            [de_lr_res, out_res, type_res] = model(LR_img.unsqueeze(0))
            sr_list.append(out_res.squeeze().cpu().numpy())
            delr_list.append(de_lr_res.squeeze().cpu().numpy())

        sr_psnr_one = 0.0
        sr_ssim_one = 0.0
        delr_psnr_one = 0.0
        delr_ssim_one = 0.0
        num = len(lr_list)
        for i in range(num):
            sr_psnr_one += calculate_psnr(gthr_list[i].numpy() * 255, sr_list[i] * 255)
            sr_ssim_one += calculate_ssim(gthr_list[i].numpy().transpose((1, 2, 0)) * 255, sr_list[i].transpose((1, 2, 0)) * 255)
            delr_psnr_one += calculate_psnr(gtlr_list[i].numpy() * 255, delr_list[i] * 255)
            delr_ssim_one += calculate_ssim(gtlr_list[i].numpy().transpose((1, 2, 0)) * 255, delr_list[i].transpose((1, 2, 0)) * 255)

        sr_psnr += sr_psnr_one/num
        sr_ssim += sr_ssim_one/num
        delr_psnr += delr_psnr_one/num
        delr_ssim += delr_ssim_one/num

        print("------------Processing {}-----------".format(index))
        print("SR_one------PSNR:{:.4f}\tSSIM:{:.4f}".format(sr_psnr_one/num, sr_ssim_one/num))
        print("deLR_one----PSNR:{:.4f}\tSSIM:{:.4f}".format(delr_psnr_one/num, delr_ssim_one/num))

        if index % 100 == 0:
            print("-------------average----------------")
            print("SR----------PSNR:{:.4f}\tSSIM:{:.4f}".format(sr_psnr / index,sr_ssim / index))
            print("deLR--------PSNR:{:.4f}\tSSIM:{:.4f}".format(delr_psnr / index, delr_ssim / index))
            print("------------------------------------")

        index += 1

        deLR = combine(delr_list, num_h, num_w, h, w, patch_size, step, 1)
        SR = combine(sr_list, num_h, num_w, h, w, patch_size, step, 4)
        #
        resultLRDeblur = transforms.ToPILImage()(torch.Tensor(deLR))
        resultLRDeblur.save(os.path.join(deLR_dir, img_name))
        resultSRDeblur = transforms.ToPILImage()(torch.Tensor(SR))
        resultSRDeblur.save(os.path.join(SR_dir,img_name))

    print("-------------average----------------")
    print("SR----------PSNR:{:.4f}\tSSIM:{:.4f}".format(sr_psnr / index, sr_ssim / index))
    print("deLR--------PSNR:{:.4f}\tSSIM:{:.4f}".format(delr_psnr / index, delr_ssim / index))
    print("------------------------------------")
