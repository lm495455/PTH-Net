from glob import glob
import os
import numpy as np

npy_path = '/root/sharedatas/LM/VideoMAEv2/get_features/MAFW/'
data_set_list = ['th14_vit_g_16_4_face', 'th14_vit_g_16_8_face', 'th14_vit_g_16_16_face',
            'th14_vit_g_16_4', 'th14_vit_g_16_8', 'th14_vit_g_16_16',
            'th14_vit_g_16_4_face_rv', 'th14_vit_g_16_8_face_rv', 'th14_vit_g_16_16_face_rv',
            'th14_vit_g_16_4_rv', 'th14_vit_g_16_8_rv', 'th14_vit_g_16_16_rv',]
txt_root = '/root/sharedatas/LM/Data/MAFW/Train & Test Set'
mode = ['compound', 'single']
mode.reverse()
txt_list = ['train.txt', 'test.txt']
labels = []
for data_set in data_set_list:
    cur_npy_path = npy_path + data_set
    npy_list = os.listdir(cur_npy_path)
    for cur_mode in mode:
        for cur_txt in txt_list:
            for i in range(1, 6):
                txt_file = os.path.join(txt_root, cur_mode, 'no_caption', 'set_' + str(i), cur_txt)
                file_data = ""
                save_file = os.path.join('MAFW', cur_mode, data_set, 'set_' + str(i) + '_' + cur_txt)
                if not os.path.exists(os.path.dirname(save_file)):
                    os.makedirs(os.path.dirname(save_file))
                with open(txt_file, "r", encoding='gb18030') as f:
                    for line in f:
                        mes = line.strip()
                        video_name, label = mes.split(' ')
                        if label not in labels:
                            labels.append(label)
                        la = labels.index(label)
                        video_name = video_name.split('.')[0]
                        if video_name + '.npy' not in npy_list:
                            continue
                        cur_np = os.path.join(cur_npy_path, video_name + '.npy')
                        feats = np.load(cur_np).astype(np.float32)
                        length = len(feats)
                        file_data += cur_np + ' ' + str(length) + ' ' + str(la) + '\n'

                with open(save_file, "w", encoding="utf-8") as f:
                    f.write(file_data)




