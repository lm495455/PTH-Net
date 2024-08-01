from glob import glob
import os
import numpy as np

npy_path = '/root/sharedatas/LM/VideoMAEv2/get_features/RAVDESS/th14_vit_g_16_8'
npy_list = os.listdir(npy_path)


all_txt_file = glob(os.path.join('/root/sharedatas/LM/Data/RAVDESS', 'set*.txt'))
new_file = 'RAVDESS/' + npy_path.split('/')[-1]
# 判断new_file文件夹是否存在，如果不存在则创建
if not os.path.exists(new_file):
    os.makedirs(new_file)


def update(file):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            video_name = line.strip()
            label = int(video_name.split("-")[2]) - 1
            if video_name + '.npy' in npy_list:
                feats = np.load(npy_path + '/' + video_name + '.npy').astype(np.float32)
                length = len(feats)
                file_data += npy_path + '/' + video_name + '.npy' + ' ' + str(length) + ' ' + str(label) + '\n'

    with open(os.path.join(new_file, file.split('/')[-1]), "w", encoding="utf-8") as f:
        f.write(file_data)


for txt_file in all_txt_file:
    update(txt_file)







