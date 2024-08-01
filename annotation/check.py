# 确保相同dataset下不同data-mode具有相同的数据
import os

# video_list_1 = os.listdir("/root/sharedatas/LM/VideoMAEv2/get_features/DFEW/th14_vit_g_16_16")
# video_list_2 = os.listdir("/root/sharedatas/LM/VideoMAEv2/get_features/DFEW/th14_vit_g_16_16_rv")
# # 找出两个list不同的元素
# diff = set(video_list_1) ^ set(video_list_2)
# print(diff)

txt_1 = 'RAVDESS/th14_vit_g_16_4/test.txt'
txt_2 = 'RAVDESS/th14_vit_g_16_4/train.txt'
video_list_1 = []
video_list_2 = []
with open(txt_1, 'r') as f:
    for line in f:
        video_name = line.strip().split(" ")[0].split('/')[-1]
        video_list_1.append(video_name)

with open(txt_2, 'r') as f:
    for line in f:
        video_name = line.strip().split(" ")[0].split('/')[-1]
        video_list_2.append(video_name)

# 查找video_list_1和video_list_2相同的部分
same = set(video_list_1) & set(video_list_2)
diff = set(video_list_1) ^ set(video_list_2)
print(same)