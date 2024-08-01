from glob import glob
import os
import numpy as np

npy_path = '/home/lab/LM/VideoMAEv2/get_features/CASME2/th14_vit_g_16_4_face'
npy_list = os.listdir(npy_path)

all_txt_file = '/home/lab/LM/Data/CASME2/Coding.txt'
new_file = 'CASME2/' + npy_path.split('/')[-1]
# 判断new_file文件夹是否存在，如果不存在则创建
if not os.path.exists(new_file):
    os.makedirs(new_file)

label_list = ['happiness', 'surprise', 'repression', 'others', 'disgust']

def update(file):
    for i in range(1, 27):
        with open(file, "r", encoding="utf-8") as f:
            train_data = ""
            test_data = ""
            for line in f:
                sub, video_name, label = line.strip().split(',')
                video_name = 'sub' + sub + '_' + video_name
                # 根据值取id
                label = label_list.index(label)
                if video_name + '.npy' in npy_list:
                    feats = np.load(npy_path + '/' + video_name + '.npy').astype(np.float32)
                    length = len(feats)
                    if int(sub) == i:
                        test_data += npy_path + '/' + video_name + '.npy' + ' ' + str(length) + ' ' + str(label) + '\n'
                    else:
                        train_data += npy_path + '/' + video_name + '.npy' + ' ' + str(length) + ' ' + str(label) + '\n'

        with open(os.path.join(new_file, 'set_' + str(i) + "_" + 'train.txt'), "w", encoding="utf-8") as f:
            f.write(train_data)
            f.close()
        with open(os.path.join(new_file, 'set_' + str(i) + "_" + 'test.txt'), "w", encoding="utf-8") as f:
            f.write(test_data)
            f.close()


update(all_txt_file)







