from glob import glob
import os
import numpy as np

npy_path = '/home/lab/LM/VideoMAEv2/get_features/MELD/train/th14_vit_g_16_16'
npy_list = os.listdir(npy_path)

all_txt_file = '/home/lab/LM/Data/MELD/Ann/train.csv'
new_file = 'MELD/' + npy_path.split('/')[-1]
# 判断new_file文件夹是否存在，如果不存在则创建
if not os.path.exists(new_file):
    os.makedirs(new_file)

label_list = ['neutral', 'surprise', 'fear', 'sadness', 'anger', 'disgust', 'joy']

def update(file):
    file_data = ""
    # data = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1, usecols=[4, 5, 7])
    with open(file, "r", encoding="utf-8") as f:
        # 读取第一行
        first_line = f.readline()
        for line in f:
            mes_list = line.strip().split(',')
            video_name = 'dia' + mes_list[-8] + '_utt' + mes_list[-7]
            # 根据值取id
            label = label_list.index(mes_list[-10])
            if video_name + '.npy' in npy_list:
                feats = np.load(npy_path + '/' + video_name + '.npy').astype(np.float32)
                length = len(feats)
                file_data += npy_path + '/' + video_name + '.npy' + ' ' + str(length) + ' ' + str(label) + '\n'

    with open(os.path.join(new_file, file.split('/')[-1]).replace('.csv', '.txt'), "w", encoding="utf-8") as f:
        f.write(file_data)


update(all_txt_file)







