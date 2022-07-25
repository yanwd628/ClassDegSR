import os.path

import numpy as np
from skimage.io import imread
import random
import torch.utils.data as data

class DataSet_ClassSR(data.Dataset):
    def __init__(self, root, istrain):
        super(DataSet_ClassSR, self).__init__()
        self.HR_path   = os.path.join(root,"HR_subs")
        self.LR_path   = os.path.join(root,"LR_subs")
        self.deLR_path = os.path.join(root,"deLR_subs",)
        self.name_list = []
        self.y = []
        class1_path = os.path.join(root+"_class", "deLR_subs", "class1")
        class2_path = os.path.join(root+"_class", "deLR_subs", "class2")
        class3_path = os.path.join(root+"_class", "deLR_subs", "class3")
        class_path = [class1_path, class2_path, class3_path]
        for i in range(len(class_path)):
            for dir in os.listdir(class_path[i]):
                self.name_list.append(dir)
                self.y.append(i)
        self.istrain = istrain

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        img_name = self.name_list[index]
        lable = self.y[index]
        HR_img = np.asarray(imread(os.path.join(self.HR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        LR_img = np.asarray(imread(os.path.join(self.LR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        deLR_img = np.asarray(imread(os.path.join(self.deLR_path, img_name)).transpose((2, 0, 1)), np.float32).copy() / 255
        if self.istrain:
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

        return deLR_img.copy(), LR_img.copy(), HR_img.copy(), lable

if __name__ == '__main__':
    myDataSet = DataSet_ClassSR("/root/yanwd/ClassDegSR/data/train_subs", True)

    index = 0
    for deLR, LR, HR, lable in myDataSet:
        print(index)
        index += 1