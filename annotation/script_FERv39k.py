from glob import glob
import os
import numpy as np

# path = '/root/sharedatas/LM/VideoMAEv2/get_features/DFEW/th14_vit_g_16_16_face_rv'
# npy_list = os.listdir(path)
# # 重命名npy_list中的所有文件
# for i in range(len(npy_list)):
#     old_name = os.path.join(path, npy_list[i])
#     new_name = os.path.join(path, str(int(npy_list[i].split('.')[0])) + '.npy')
#     os.rename(old_name, new_name)

your_dataset_path = ""
all_txt_file = glob(os.path.join('FERv39k/*_All.txt'))

npy_path = '/root/sharedatas/LM/VideoMAEv2/get_features/FERv39k/th14_vit_g_16_4_face_rv'
# npy_list = os.listdir(npy_path)
new_file = 'FERv39k/' + npy_path.split('/')[-1]
# 判断new_file文件夹是否存在，如果不存在则创建
if not os.path.exists(new_file):
    os.makedirs(new_file)


def update(file):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            video_name = line.strip().split(" ")[0].split('FERV39k/')[-1]
            # if video_name + '.npy' in npy_list:
            feats = np.load(npy_path + '/' + video_name + '.npy').astype(np.float32)
            length = len(feats)
            file_data += npy_path + '/' + video_name + '.npy' + ' ' + str(length) + ' ' + line.strip().split(" ")[-1] + '\n'

    with open(os.path.join(new_file, file), "w", encoding="utf-8") as f:
        f.write(file_data)


for txt_file in all_txt_file:
    update(txt_file)




