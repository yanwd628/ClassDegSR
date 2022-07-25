import shutil
import os

source_list = ["/data/yanwd/ClassDegSR/data/test_subs/HR_subs",
               "/data/yanwd/ClassDegSR/data/test_subs/LR_subs",
               "/data/yanwd/ClassDegSR/data/test_subs/deHR_subs",
               "/data/yanwd/ClassDegSR/data/test_subs/deLR_subs"]

target_list = ["/data/yanwd/ClassDegSR/data/test_subs_class/HR_subs/class1",
               "/data/yanwd/ClassDegSR/data/test_subs_class/HR_subs/class2",
               "/data/yanwd/ClassDegSR/data/test_subs_class/HR_subs/class3",
               "/data/yanwd/ClassDegSR/data/test_subs_class/LR_subs/class1",
               "/data/yanwd/ClassDegSR/data/test_subs_class/LR_subs/class2",
               "/data/yanwd/ClassDegSR/data/test_subs_class/LR_subs/class3",
               "/data/yanwd/ClassDegSR/data/test_subs_class/deHR_subs/class1",
               "/data/yanwd/ClassDegSR/data/test_subs_class/deHR_subs/class2",
               "/data/yanwd/ClassDegSR/data/test_subs_class/deHR_subs/class3",
               "/data/yanwd/ClassDegSR/data/test_subs_class/deLR_subs/class1",
               "/data/yanwd/ClassDegSR/data/test_subs_class/deLR_subs/class2",
               "/data/yanwd/ClassDegSR/data/test_subs_class/deLR_subs/class3"]

threshold=[23.6624,28.5995]

res_path = "/data/yanwd/ClassDegSR/data/test_LR_deLR_res.txt"

file = open(res_path, 'r')
data_list = file.readlines()

index = 0
for data in data_list:
    index += 1
    print(index)
    data = data.split('\t')
    img_name, psnr = data[1].rstrip()+".png", float(data[2])
    if psnr < threshold[0]:  #hard  class1
        shutil.copy(os.path.join(source_list[0], img_name), os.path.join(target_list[0], img_name))
        shutil.copy(os.path.join(source_list[1], img_name), os.path.join(target_list[3], img_name))
        shutil.copy(os.path.join(source_list[2], img_name), os.path.join(target_list[6], img_name))
        shutil.copy(os.path.join(source_list[3], img_name), os.path.join(target_list[9], img_name))
    elif psnr >= threshold[0] and psnr < threshold[1]: #middle class2
        shutil.copy(os.path.join(source_list[0], img_name), os.path.join(target_list[1], img_name))
        shutil.copy(os.path.join(source_list[1], img_name), os.path.join(target_list[4], img_name))
        shutil.copy(os.path.join(source_list[2], img_name), os.path.join(target_list[7], img_name))
        shutil.copy(os.path.join(source_list[3], img_name), os.path.join(target_list[10], img_name))
    else:    #easy class3
        shutil.copy(os.path.join(source_list[0], img_name), os.path.join(target_list[2], img_name))
        shutil.copy(os.path.join(source_list[1], img_name), os.path.join(target_list[5], img_name))
        shutil.copy(os.path.join(source_list[2], img_name), os.path.join(target_list[8], img_name))
        shutil.copy(os.path.join(source_list[3], img_name), os.path.join(target_list[11], img_name))


file.close()
