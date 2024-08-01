# import os
# root_path = 'RAVDESS/Random'
# # 只取root_path下的文件夹
# feature_list = os.listdir(root_path)
# for cur_feature in feature_list:
#     txt_list = os.listdir(os.path.join(root_path, cur_feature))
#     for cur_txt in txt_list:
#         new_txt = ''
#         with open(os.path.join(root_path, cur_feature, cur_txt), "r", encoding="utf-8") as f:
#             for line in f:
#                 line = 'get_features' + line.split('get_features')[-1]
#                 new_txt += line
#         with open(os.path.join(root_path, cur_feature, cur_txt), "w", encoding="utf-8") as f:
#             f.write(new_txt)
#         print(cur_feature + ' ' + cur_txt)