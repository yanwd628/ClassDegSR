'''Extract subimages for HR and deHR'''

import os
import sys
import cv2
import numpy as np

def main():
    root_folder   = '/data/yanwd/ClassDegSR/data/test'

    save_HR_subs_folder   = '/data/yanwd/ClassDegSR/data/test_subs/HR_subs'
    save_deHR_subs_folder = '/data/yanwd/ClassDegSR/data/test_subs/deHR_subs'

    save_LR_subs_folder   = '/data/yanwd/ClassDegSR/data/test_subs/LR_subs'
    save_deLR_subs_folder = '/data/yanwd/ClassDegSR/data/test_subs/deLR_subs'

    # mkdir(save_deLR_subs_folder)
    # mkdir(save_LR_subs_folder)
    # mkdir(save_HR_subs_folder)
    # mkdir(save_deHR_subs_folder)
    train = False

    compression_level = 3
    scale_ratio = 4
    if train:
        crop_sz = 256
        step = 128
        thres_sz = 32
        downsizes = [0.5,0.7,1]
    else:
        crop_sz = 256
        step = 256
        thres_sz = 0
        downsizes = [1]


    num = 1
    root_list = os.listdir(root_folder)
    for path in root_list:
        cur_path  = os.path.join(root_folder, path)
        HR_path   = os.path.join(cur_path, 'sharp')
        deHR_path = os.path.join(cur_path, 'blur_gamma')
        img_list   = os.listdir(HR_path)
        for img_name in img_list:
            HR_img   = cv2.imread(os.path.join(HR_path, img_name))
            deHR_img = cv2.imread(os.path.join(deHR_path, img_name))
            n_channels = len(HR_img.shape)
            if n_channels == 2:
                h, w = HR_img.shape
            elif n_channels == 3:
                h, w, c = HR_img.shape
            else:
                raise ValueError('Wrong image shape - {}'.format(n_channels))

            for downsize in downsizes:
                down_h = round(h * downsize)
                down_w = round(w * downsize)
                down_HR_img   = cv2.resize(HR_img, (down_w, down_h), interpolation=cv2.INTER_CUBIC)
                down_deHR_img = cv2.resize(deHR_img, (down_w, down_h), interpolation=cv2.INTER_CUBIC)

                h_space = np.arange(0, down_h - crop_sz + 1, step)
                if down_h - (h_space[-1] + crop_sz) > thres_sz:
                    h_space = np.append(h_space, down_h - crop_sz)
                w_space = np.arange(0, down_w - crop_sz + 1, step)
                if down_w - (w_space[-1] + crop_sz) > thres_sz:
                    w_space = np.append(w_space, down_w - crop_sz)


                for type in ['HR', 'deHR']:
                    index = 0
                    if type=='HR':
                        img = down_HR_img
                        save_subs_folder = save_HR_subs_folder
                        save_down_subs_folder = save_LR_subs_folder
                    else:
                        img = down_deHR_img
                        save_subs_folder = save_deHR_subs_folder
                        save_down_subs_folder = save_deLR_subs_folder

                    for x in h_space:
                        for y in w_space:
                            index += 1
                            if n_channels == 2:
                                crop_img = img[x:x + crop_sz, y:y + crop_sz]
                                d_h, d_w = crop_img.shape
                                d_w, d_h = d_w//scale_ratio, d_h//scale_ratio
                                down_crop_img = cv2.resize(crop_img, (d_w, d_h), interpolation=cv2.INTER_CUBIC)
                            else:
                                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                                d_h, d_w, d_c = crop_img.shape
                                d_w, d_h = d_w // scale_ratio, d_h // scale_ratio
                                down_crop_img = cv2.resize(crop_img, (d_w, d_h), interpolation=cv2.INTER_CUBIC)

                            crop_img = np.ascontiguousarray(crop_img)
                            down_crop_img = np.ascontiguousarray(down_crop_img)
                            cv2.imwrite(
                                os.path.join(save_subs_folder,'{:04d}_s{:03d}.png'.format(num, index)),crop_img,
                                [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                            cv2.imwrite(
                                os.path.join(save_down_subs_folder, '{:04d}_s{:03d}.png'.format(num, index)), down_crop_img,
                                [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                    print("Processing {:s}_{}_{}".format(os.path.join(HR_path, img_name), downsize, type))
                num +=1


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('mkdir [{:s}] ...'.format(path))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(path))
        sys.exit(1)


if __name__ == '__main__':
    main()