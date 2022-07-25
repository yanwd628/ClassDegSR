import os.path
from skimage.io import imread
import torch.utils.data as data
import numpy as np
import random


class dataset_class(data.Dataset):
    def __init__(self, root, istrain):
        super(dataset_class, self).__init__()
        self.x = []
        self.y = []
        class1_path = os.path.join(root, "class1")
        class2_path = os.path.join(root, "class2")
        class3_path = os.path.join(root, "class3")
        class_path = [class1_path, class2_path, class3_path]
        for i in range(len(class_path)):
            for dir in os.listdir(class_path[i]):
                self.x.append(os.path.join(class_path[i],dir))
                self.y.append(i)
        self.istrain = istrain

    def __getitem__(self, index):
        image_path = self.x[index]
        image = np.asarray(imread(image_path).transpose((2, 0, 1)), np.float32).copy() / 255
        lable = self.y[index]
        if self.istrain:
            if random.randint(0, 1) == 0:
                image = np.flip(image, 2)
            # randomly rotation
            rotation_times = random.randint(0, 3)
            image = np.rot90(image, rotation_times, (1, 2))

        return image.copy(), lable

    def __len__(self):
        return len(self.y)

if __name__ == '__main__':
    root = "/data/yanwd/ClassDegSR/data/train_subs_class/deLR_subs"
    dataset = dataset_class(root, True)
    print(len(dataset))

