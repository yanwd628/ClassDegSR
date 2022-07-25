import os.path

import numpy as np
from skimage.io import imread
import random
import torch.utils.data as data

class DataSet(data.Dataset):
    def __init__(self, root, type):
        super(DataSet, self).__init__()
        self.HR_path   = os.path.join(root,"HR_subs",type)
        self.LR_path   = os.path.join(root,"LR_subs",type)
        self.deLR_path = os.path.join(root,"deLR_subs",type)
        self.name_list = os.listdir(os.path.join(root,"HR_subs",type))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        img_name = self.name_list[index]
        HR_img = np.asarray(imread(os.path.join(self.HR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        LR_img = np.asarray(imread(os.path.join(self.LR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        deLR_img = np.asarray(imread(os.path.join(self.deLR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        # randomly flip
        if random.randint(0, 1) == 0:
            HR_img   = np.flip(HR_img, 2)
            LR_img   = np.flip(LR_img, 2)
            deLR_img = np.flip(deLR_img, 2)
        # randomly rotation
        rotation_times = random.randint(0, 3)
        HR_img   = np.rot90(HR_img, rotation_times, (1, 2))
        LR_img   = np.rot90(LR_img, rotation_times, (1, 2))
        deLR_img = np.rot90(deLR_img, rotation_times, (1, 2))

        return deLR_img.copy(), LR_img.copy(), HR_img.copy()

class DataTestSet(data.Dataset):
    def __init__(self, root, type):
        super(DataTestSet, self).__init__()
        self.HR_path   = os.path.join(root,"HR_subs",type)
        self.LR_path   = os.path.join(root,"LR_subs",type)
        self.deLR_path = os.path.join(root,"deLR_subs",type)

        self.name_list = os.listdir(os.path.join(root,"HR_subs",type))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        img_name = self.name_list[index]
        HR_img = np.asarray(imread(os.path.join(self.HR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        LR_img = np.asarray(imread(os.path.join(self.LR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        deLR_img = np.asarray(imread(os.path.join(self.deLR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255

        return deLR_img.copy(), LR_img.copy(), HR_img.copy(), img_name



if __name__ == '__main__':
    myDataSet = DataTestSet("/root/yanwd/2022/classDeSR/data/test_subs/Class", "class1")
    for a,b,c,d in myDataSet:
        print(d.type)
        break
