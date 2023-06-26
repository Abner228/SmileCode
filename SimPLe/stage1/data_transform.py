import numpy as np
import torch

class Norm(object):
    def __call__(self, sample):
        img_data = sample['image']
        mean = np.mean(img_data)
        std = np.std(img_data)
        img_data = (img_data - mean) / std
        sample['image'] = img_data
        return sample

class RandomCrop(object):
    def __init__(self, CropSize, seed, fg_rate):
        self.CropSize = CropSize
        self.seed = seed
        self.fg_rate = fg_rate

    def __call__(self, sample):
        img_data, seg_data, gt_data = sample['image'], sample['label'], sample['gt']

        '''Padding if necessary'''
        if seg_data.shape[0] <= self.CropSize[0] or seg_data.shape[1] <= self.CropSize[1] or seg_data.shape[2] <= \
                self.CropSize[2]:
            pw = max((self.CropSize[0] - seg_data.shape[0]) // 2 + 3, 0)
            ph = max((self.CropSize[1] - seg_data.shape[1]) // 2 + 3, 0)
            pd = max((self.CropSize[2] - seg_data.shape[2]) // 2 + 3, 0)
            img_data = np.pad(img_data, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            seg_data = np.pad(seg_data, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            gt_data = np.pad(gt_data, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        if np.random.random() < self.fg_rate:
            '''Foreground patch'''
            all_locs = np.argwhere(seg_data == self.seed)
            selected_voxel = all_locs[np.random.choice(len(all_locs))]
            w1_lb = max(0, selected_voxel[0] - self.CropSize[0] // 2)
            h1_lb = max(0, selected_voxel[1] - self.CropSize[1] // 2)
            d1_lb = max(0, selected_voxel[2] - self.CropSize[2] // 2)

            w1_ub = min(w1_lb + self.CropSize[0], seg_data.shape[0])
            h1_ub = min(h1_lb + self.CropSize[1], seg_data.shape[1])
            d1_ub = min(d1_lb + self.CropSize[2], seg_data.shape[2])

            seg_data_crop = seg_data[w1_ub - self.CropSize[0]:w1_ub, h1_ub - self.CropSize[1]:h1_ub, d1_ub - self.CropSize[2]:d1_ub]
            img_data_crop = img_data[w1_ub - self.CropSize[0]:w1_ub, h1_ub - self.CropSize[1]:h1_ub, d1_ub - self.CropSize[2]:d1_ub]
            gt_data_crop = gt_data[w1_ub - self.CropSize[0]:w1_ub, h1_ub - self.CropSize[1]:h1_ub, d1_ub - self.CropSize[2]:d1_ub]
        else:
            '''Random patch'''
            (w, h, d) = img_data.shape
            w1 = np.random.randint(0, w - self.CropSize[0])
            h1 = np.random.randint(0, h - self.CropSize[1])
            d1 = np.random.randint(0, d - self.CropSize[2])
            seg_data_crop = seg_data[w1:w1 + self.CropSize[0], h1:h1 + self.CropSize[1], d1:d1 + self.CropSize[2]]
            img_data_crop = img_data[w1:w1 + self.CropSize[0], h1:h1 + self.CropSize[1], d1:d1 + self.CropSize[2]]
            gt_data_crop = gt_data[w1:w1 + self.CropSize[0], h1:h1 + self.CropSize[1], d1:d1 + self.CropSize[2]]

        sample['image'], sample['label'], sample['gt'] = img_data_crop, seg_data_crop, gt_data_crop
        return sample

class ToTensor(object):
    def __init__(self, channel_axis):
        self.channel_axis = channel_axis

    def __call__(self, sample):
        img_data, seg_data = sample['image'], sample['label']
        img_data = torch.Tensor(np.expand_dims(img_data, axis=self.channel_axis).copy())
        seg_data = seg_data.astype('uint8')
        seg_data = torch.LongTensor(seg_data.copy())
        sample['image'], sample['label'] = img_data, seg_data

        return sample