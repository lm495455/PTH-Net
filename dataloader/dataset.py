from numpy.random import randint
from torch.utils import data
from dataloader.video_transform import *
import numpy as np


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, dataset='DFEW', data_set=1, max_len=16, mode='train', data_mode='norm', is_face=True, label_type='single'):
        self.file_path = "./annotation/" + dataset + '/'
        if dataset == 'MAFW':
            self.file_path += label_type + '/'
        if is_face:
            if data_mode == 'norm':
                self.data_path = '_face'
            elif data_mode == 'rv':
                self.data_path = '_face_rv'
            else:
                self.data_path = '_face_flow'
        else:
            if data_mode == 'norm':
                self.data_path = ''
            elif data_mode == 'rv':
                self.data_path = '_rv'
            else:
                self.data_path = '_flow'
        if dataset in ['DFEW', 'MAFW', 'RAVDESS', 'CREMA-D', 'eNTERFACE05', 'CASME2']:
            list_file = "set_" + str(data_set) + "_" + mode + ".txt"
        elif dataset == 'FERv39k':
            list_file = mode + "_All" + ".txt"
        else:
            list_file = mode + ".txt"
        file_name = ['th14_vit_g_16_4', 'th14_vit_g_16_8', 'th14_vit_g_16_16']
        # file_name = ['th14_vit_g_16_2', 'th14_vit_g_16_4', 'th14_vit_g_16_8']
        self.list_file = [self.file_path + file_name[0] + self.data_path + '/' + list_file,
                          self.file_path + file_name[1] + self.data_path + '/' + list_file,
                          self.file_path + file_name[2] + self.data_path + '/' + list_file]
        self.max_len = max_len
        self.mode = mode
        self.crop_ratio = [0.9, 1.0]
        self.input_noise = 0.0005
        self.num_frames_H = self.max_len
        self.num_frames_M = int(self.max_len / 2)
        self.num_frames_L = int(self.max_len / 4)
        self._parse_list()
        pass

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        tmp_H = [x.strip().split(' ') for x in open(self.list_file[0])]
        tmp_M = [x.strip().split(' ') for x in open(self.list_file[1])]
        tmp_L = [x.strip().split(' ') for x in open(self.list_file[2])]

        self.video_list_H = [VideoRecord(item) for item in tmp_H]
        self.video_list_M = [VideoRecord(item) for item in tmp_M]
        self.video_list_L = [VideoRecord(item) for item in tmp_L]
        print(('video number:%d' % (len(self.video_list_H))))

    def _get_seq_frames(self, record, NUM_FRAMES):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
            temporal_sample_index (int): temporal sample index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = NUM_FRAMES
        video_length = record.num_frames
        if video_length < num_frames:
            return [0]
        else:
            seg_size = float(video_length) / num_frames
            seq = []
            # index from 1, must add 1
            if self.mode == "train":
                for i in range(num_frames):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    seq.append(randint(start, end))
            else:
                duration = seg_size / 2  # 取中间那帧
                for i in range(num_frames):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    frame_index = start + int(duration)
                    seq.append(frame_index)
            return seq

    def __getitem__(self, index):
        record_H = self.video_list_H[index]
        record_M = self.video_list_M[index]
        record_L = self.video_list_L[index]
        if self.mode == 'train':
            segment_indices_H = self._get_seq_frames(record_H, self.num_frames_H)
            segment_indices_M = self._get_seq_frames(record_M, self.num_frames_M)
            segment_indices_L = self._get_seq_frames(record_L, self.num_frames_L)
        elif self.mode == 'test':
            segment_indices_H = self._get_seq_frames(record_H, self.num_frames_H)
            segment_indices_M = self._get_seq_frames(record_M, self.num_frames_M)
            segment_indices_L = self._get_seq_frames(record_L, self.num_frames_L)
        return self.get(record_H, segment_indices_H, 1), self.get(record_M, segment_indices_M, 2), self.get(record_L,
                                                                                                            segment_indices_L,
                                                                                                            4), record_H.label

    """按照特征长度来选择H，M，L"""

    # def __getitem__(self, index):
    #     record_H = self.video_list_H[index]
    #     record_M = self.video_list_M[index]
    #     record_L = self.video_list_L[index]
    #     if self.mode == 'train':
    #         segment_indices_H = self._get_seq_frames(record_H, 16)
    #         segment_indices_M = self._get_seq_frames(record_M, 8)
    #         segment_indices_L = self._get_seq_frames(record_L, 4)
    #         return self.get(record_H, segment_indices_H, 1), self.get(record_M, segment_indices_M, 2), self.get(
    #             record_L,
    #             segment_indices_L,
    #             4), record_H.label
    #     elif self.mode == 'test':
    #         if record_H.num_frames < 16:
    #             segment_indices_L = self._get_seq_frames(record_H, 4)
    #             return None, None, None, None, self.get(
    #                 record_H,
    #                 segment_indices_L,
    #                 4), record_H.label
    #         elif record_H.num_frames < 32:
    #             segment_indices_M = self._get_seq_frames(record_H, 8)
    #             return None, None, None, None, self.get(
    #                 record_H,
    #                 segment_indices_M,
    #                 2), record_H.label
    #         else:
    #             segment_indices_H = self._get_seq_frames(record_H, 16)
    #             return None, None, None, None, self.get(
    #                 record_H,
    #                 segment_indices_H,
    #                 1), record_H.label

    def get(self, record, indices, n, padding_val=0.0):
        video_item = record.path
        feats = np.load(video_item).astype(np.float32)
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        result = []
        if len(indices) == 1:
            result_feats = feats
        else:
            for seg_ind in indices:
                p = int(seg_ind)
                result.append(feats[:, p:p + 1])
            # 将result_feats中的tensor组成一个tensor
            result_feats = torch.cat(result, dim=1)
        cur_len = result_feats.shape[-1]
        batch_shape = [result_feats.shape[0], int(self.max_len / n)]
        batched_inputs = result_feats.new_full(batch_shape, padding_val)
        batched_inputs[:, :result_feats.shape[-1]].copy_(result_feats)
        if self.mode == 'train' and self.input_noise > 0:
            noise = torch.randn_like(batched_inputs) * self.input_noise
            batched_inputs += noise

        batched_masks = torch.arange(int(self.max_len / n))[None, :] < cur_len
        return batched_inputs, batched_masks

    def __len__(self):
        return len(self.video_list_H)


def train_data_loader(dataset, data_mode='norm', data_set=None, is_face=True, label_type='single'):
    train_data = VideoDataset(dataset=dataset,
                              data_set=data_set,
                              max_len=16,
                              mode='train',
                              data_mode=data_mode,
                              is_face=is_face,
                              label_type=label_type)
    return train_data


def test_data_loader(dataset, data_mode='norm', data_set=None, is_face=True, label_type='single'):
    test_data = VideoDataset(dataset=dataset,
                             data_set=data_set,
                             max_len=16,
                             mode='test',
                             data_mode=data_mode,
                             is_face=is_face,
                             label_type=label_type)
    return test_data
