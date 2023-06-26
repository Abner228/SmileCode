import os
from data import BreastTumor
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage import transform
import SimpleITK as sitk
import nibabel as nib
from nibabel import io_orientation
from data_transform import Norm, RandomCrop, ToTensor
import sys
import math


sys.path.append('..')
from net import Unet
from file_and_folder_operations import read_data_list, myMakedirs, getSubFiles


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
patch_size = [96, 128, 128]
num_classes = 2
train_data_list = os.listdir('/data/zym/workspace/noisy/noisy_label_resample_1.0X0.6X0.6')
train_dataset = BreastTumor(train_data_list, transform=Norm())
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

net = Unet(1, 2).cuda()
net.load_state_dict(torch.load('/data/zym/experiment/noisy/stage2/iter_100.pth'))

iter_label_dir = '/data/zym/experiment/noisy/stage2/update_mask'
myMakedirs(iter_label_dir, overwrite=False)

for sampled_batch in tqdm(train_dataloader):
    net.eval()
    case, img, seg = sampled_batch['case'], sampled_batch['image'], sampled_batch['label']
    img = np.squeeze(img.numpy(), axis=0)
    seg = np.squeeze(seg.numpy(), axis=0)
    seg[seg == 2] = 1   # 1 - inside bbox, 0 - outside bbox

    w, h, d = img.shape
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        img = np.pad(img, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = img.shape

    stride_x = int(patch_size[0] / 2)
    stride_y = int(patch_size[1] / 2)
    stride_z = int(patch_size[2] / 2)
    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + img.shape).astype(np.float32)
    cnt = np.zeros(img.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_x * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = img[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=(0, 1)).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                with torch.no_grad():
                    pred, _ = net(test_patch)
                    y1 = F.softmax(pred, dim=1)
                y1 = y1.cpu().data.numpy()

                y1 = y1[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y1
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1.
    score_map = score_map / np.expand_dims(cnt, axis=0)
    if add_pad:
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    label_map = np.argmax(score_map, axis=0)
    label_map = (seg * label_map).astype(np.uint8)

    img_npz = '/data/zym/YN/DCE/resample_1.0X0.6X0.6/DCE_C1/' + case[0] + '.npz'
    img_data = np.load(img_npz)

    pred_path = os.path.join(iter_label_dir, case[0] + '_pred.nii.gz')
    seg_zoom = transform.resize(label_map.astype(float), img_data['origin_size'],
                                0, mode="edge", clip=True,
                                anti_aliasing=False).astype(np.uint8)
    output = sitk.GetImageFromArray(seg_zoom)
    sitk.WriteImage(output, pred_path.replace('.nii.gz', '_reorientation.nii.gz'))
    img = nib.load(pred_path.replace('.nii.gz', '_reorientation.nii.gz'))
    img = img.as_reoriented(io_orientation(img_data['origin_affine']))
    nib.save(img, pred_path)

    seg = sitk.ReadImage(pred_path)
    seg.SetSpacing(img_data['origin_spacing'][[2, 1, 0]])
    seg.SetDirection(img_data['origin_direction'])
    seg.SetOrigin(img_data['origin'])
    sitk.WriteImage(seg, pred_path)
    os.remove(pred_path.replace('.nii.gz', '_reorientation.nii.gz'))



