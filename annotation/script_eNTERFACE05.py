from glob import glob
import os
import numpy as np

npy_path = '/root/sharedatas/LM/VideoMAEv2/get_features/eNTERFACE05/th14_vit_g_16_16_rv'
npy_list = os.listdir(npy_path)


all_txt_file = glob(os.path.join('/root/sharedatas/LM/Data/eNTERFACE05', 'set*.txt'))
new_file = 'eNTERFACE05/' + npy_path.split('/')[-1]
# 判断new_file文件夹是否存在，如果不存在则创建
if not os.path.exists(new_file):
    os.makedirs(new_file)

label_list = ['an', 'ha', 'su', 'fe', 'sa', 'di']
def update(file):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            video_name = line.strip()
            label = video_name.split('_')[1]
            label = label_list.index(label)
            if video_name + '.npy' not in npy_list:
                continue
            feats = np.load(npy_path + '/' + video_name + '.npy').astype(np.float32)
            length = len(feats)
            file_data += npy_path + '/' + video_name + '.npy' + ' ' + str(length) + ' ' + str(label) + '\n'

    with open(os.path.join(new_file, file.split('/')[-1]), "w", encoding="utf-8") as f:
        f.write(file_data)


for txt_file in all_txt_file:
    update(txt_file)



