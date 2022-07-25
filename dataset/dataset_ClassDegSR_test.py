import os.path

import numpy as np
from skimage.io import imread,imsave
from skimage.transform import rescale
import random
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt

class DataSet_ClassDegSR_test(data.Dataset):
    def __init__(self, root):
        super(DataSet_ClassDegSR_test, self).__init__()
        self.HR_path   = os.path.join(root,"HR")
        self.deLR_path = os.path.join(root,"LR_Blur",)
        self.LR_path   = os.path.join(root,"LR")
        self.name_list = os.listdir(self.HR_path)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        img_name = self.name_list[index]

        HR_img = np.asarray(imread(os.path.join(self.HR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        deLR_img = np.asarray(imread(os.path.join(self.deLR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        LR_img = np.asarray(imread(os.path.join(self.LR_path, img_name)).transpose((2, 0, 1)),np.float32).copy() / 255

        return deLR_img.copy(), LR_img.copy(), HR_img.copy(), img_name

if __name__ == '__main__':
    myDataSet = DataSet_ClassDegSR_test("/root/yanwd/ClassDegSR/data/Test_4x")

    index = 0
    for deLR, LR, HR, name in myDataSet:
        print(deLR.shape)
        print(len(deLR.shape))
        break
