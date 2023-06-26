import SimpleITK as sitk
import numpy as np
import os
import dijkstra3d
from scipy import ndimage
from skimage.segmentation import random_walker

img_sitk = sitk.ReadImage('data/C1.nii.gz')
ep_sitk = sitk.ReadImage('data/ep.nii.gz')

img = sitk.GetArrayFromImage(img_sitk)
ep = sitk.GetArrayFromImage(ep_sitk)
l0, l1, l2 = img.shape

# find 6 points
d0_mn = np.where(ep == 1)[0][0]
d0_mx = np.where(ep == 2)[0][0]
d1_mn = np.where(ep == 3)[1][0]
d1_mx = np.where(ep == 4)[1][0]
d2_mn = np.where(ep == 5)[2][0]
d2_mx = np.where(ep == 6)[2][0]

# extend bbox
d0_mn_extend = d0_mn - 10
d1_mn_extend = d1_mn - 20
d2_mn_extend = d2_mn - 20
if d0_mn_extend < 0:
    d0_mn_extend = 0
if d1_mn_extend < 0:
    d1_mn_extend = 0
if d2_mn_extend < 0:
    d2_mn_extend = 0

d0_mx_extend = d0_mx + 10
d1_mx_extend = d1_mx + 20
d2_mx_extend = d2_mx + 20
if d0_mx_extend > l0 - 1:
    d0_mx_extend = l0 - 1
if d1_mx_extend > l1 - 1:
    d1_mx_extend = l1 - 1
if d2_mx_extend > l2 - 1:
    d2_mx_extend = l2 - 1

# crop
img_crop = img[d0_mn_extend:d0_mx_extend, d1_mn_extend:d1_mx_extend, d2_mn_extend:d2_mx_extend]
ep_crop = ep[d0_mn_extend:d0_mx_extend, d1_mn_extend:d1_mx_extend, d2_mn_extend:d2_mx_extend]

# crop img gradient
img_crop_sitk = sitk.GetImageFromArray(img_crop)
img_crop_sitk_float = sitk.Cast(img_crop_sitk, sitk.sitkFloat32)
sobel_op = sitk.SobelEdgeDetectionImageFilter()
sobel_sitk = sobel_op.Execute(img_crop_sitk_float)
sobel_sitk.SetSpacing(img_sitk.GetSpacing())
sobel_sitk.SetDirection(img_sitk.GetDirection())
grad_crop = sitk.GetArrayFromImage(sobel_sitk)

#  search the shortest path on gradient
scribble_crop = np.zeros_like(img_crop)
source = (np.where(ep_crop == 1)[0][0], np.where(ep_crop == 1)[1][0], np.where(ep_crop == 1)[2][0])
target = (np.where(ep_crop == 2)[0][0], np.where(ep_crop == 2)[1][0], np.where(ep_crop == 2)[2][0])
path = dijkstra3d.dijkstra(grad_crop, source, target, connectivity=26)
for i in range(path.size // 3):
    scribble_crop[path[i, 0], path[i, 1], path[i, 2]] = 1

source = (np.where(ep_crop == 3)[0][0], np.where(ep_crop == 3)[1][0], np.where(ep_crop == 3)[2][0])
target = (np.where(ep_crop == 4)[0][0], np.where(ep_crop == 4)[1][0], np.where(ep_crop == 4)[2][0])
path = dijkstra3d.dijkstra(grad_crop, source, target, connectivity=26)
for i in range(path.size // 3):
    scribble_crop[path[i, 0], path[i, 1], path[i, 2]] = 1

source = (np.where(ep_crop == 5)[0][0], np.where(ep_crop == 5)[1][0], np.where(ep_crop == 5)[2][0])
target = (np.where(ep_crop == 6)[0][0], np.where(ep_crop == 6)[1][0], np.where(ep_crop == 6)[2][0])
path = dijkstra3d.dijkstra(grad_crop, source, target, connectivity=26)
for i in range(path.size // 3):
    scribble_crop[path[i, 0], path[i, 1], path[i, 2]] = 1

s = ndimage.generate_binary_structure(3, 3)
scribble_crop = ndimage.binary_dilation(scribble_crop, structure=s).astype(scribble_crop.dtype)

# save scribble
# scribble = np.zeros_like(img)
# scribble[d0_mn_extend:d0_mx_extend, d1_mn_extend:d1_mx_extend, d2_mn_extend:d2_mx_extend] = scribble_crop
# scribble_sitk = sitk.GetImageFromArray(scribble)
# scribble_sitk.SetSpacing(img_sitk.GetSpacing())
# scribble_sitk.SetDirection(img_sitk.GetDirection())
# scribble_sitk.SetOrigin(img_sitk.GetOrigin())
# sitk.WriteImage(scribble_sitk, 'data/scribble.nii.gz')

d0_mn = np.where(ep_crop == 1)[0][0]
d0_mx = np.where(ep_crop == 2)[0][0]
d1_mn = np.where(ep_crop == 3)[1][0]
d1_mx = np.where(ep_crop == 4)[1][0]
d2_mn = np.where(ep_crop == 5)[2][0]
d2_mx = np.where(ep_crop == 6)[2][0]

# bg seed
scribble_crop[:d0_mn - 5, :, :] = 2
scribble_crop[d0_mx + 5:, :, :] = 2
scribble_crop[:, :d1_mn - 5, :] = 2
scribble_crop[:, d1_mx + 5:, :] = 2
scribble_crop[:, :, :d2_mn - 5] = 2
scribble_crop[:, :, d2_mx + 5:] = 2

img_crop_norm = (img_crop - np.min(img_crop)) / np.max(img_crop)
spacing = np.array(img_sitk.GetSpacing())[[2, 1, 0]]
labels_crop = random_walker(img_crop_norm, scribble_crop, beta=130, mode='cg_mg', return_full_prob=True, spacing=spacing)
labels_crop = labels_crop[0]
scribble_crop = np.zeros_like(labels_crop)
scribble_crop[labels_crop > 0.8] = 1
scribble_crop[labels_crop <= 0.1] = 2

times = 7
for i in range(times):
    labels_crop = random_walker(img_crop_norm, scribble_crop, beta=130, mode='cg_mg', return_full_prob=True, spacing=spacing)
    labels_crop = labels_crop[0]
    if i < times:
        scribble_crop = np.zeros_like(labels_crop)
        scribble_crop[labels_crop > 0.8] = 1
        scribble_crop[labels_crop <= 0.1] = 2

labels_crop[labels_crop > 0.8] = 1
labels_crop[labels_crop <= 0.8] = 0

labels_crop_box = labels_crop[d0_mn:d0_mx+1, d1_mn:d1_mx+1, d2_mn:d2_mx+1]
labels_crop_box[labels_crop_box == 0] = 2
labels_crop[d0_mn:d0_mx+1, d1_mn:d1_mx+1, d2_mn:d2_mx+1] = labels_crop_box
labels_crop[:d0_mn, :, :] = 0
labels_crop[d0_mx+1:, :, :] = 0
labels_crop[:, :d1_mn, :] = 0
labels_crop[:, d1_mx+1:, :] = 0
labels_crop[:, :, :d2_mn] = 0
labels_crop[:, :, d2_mx+1:] = 0

labels = np.zeros_like(img)
labels[d0_mn_extend:d0_mx_extend, d1_mn_extend:d1_mx_extend, d2_mn_extend:d2_mx_extend] = labels_crop
labels_sitk = sitk.GetImageFromArray(labels)
labels_sitk.SetSpacing(img_sitk.GetSpacing())
labels_sitk.SetDirection(img_sitk.GetDirection())
labels_sitk.SetOrigin(img_sitk.GetOrigin())
sitk.WriteImage(labels_sitk, 'data/init_mask.nii.gz')

# 0 - bg, 1 - fg, 2 - unlabeled