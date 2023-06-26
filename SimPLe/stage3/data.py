from torch.utils.data import Dataset
import numpy as np

class BreastTumor(Dataset):
    def __init__(self, info_list, transform=None):
        self.info_list = info_list
        self.transform = transform

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        case = self.info_list[idx]
        seg_npz = '/data/zym/workspace/noisy/stage2/update_label_resample_1.0X0.6X0.6/' + case
        img_npz = '/data/zym/YN/DCE/resample_1.0X0.6X0.6/DCE_C1/' + case
        gt_npz = '/data/zym/YN/DCE/resample_1.0X0.6X0.6/TumorMask/' + case
        case = case.split('.')[0]
        seg_data = np.load(seg_npz)
        img_data = np.load(img_npz)
        gt_data = np.load(gt_npz)

        img = img_data['data']
        seg = seg_data['seg']
        gt = gt_data['seg']

        sample = {'case': case,
                  'image': img,
                  'label': seg,
                  'gt': gt}
        if self.transform:
            sample = self.transform(sample)
        return sample

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def next(self):
        sample = self.sample
        self.preload()
        return sample

    def preload(self):
        try:
            self.sample = next(self.loader)
        except StopIteration:
            self.sample = None
            return