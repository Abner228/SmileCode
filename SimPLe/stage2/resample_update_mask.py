import SimpleITK as sitk
import os
from multiprocessing import Pool
import nibabel as nib
import numpy as np
from nibabel import io_orientation
from skimage import transform

import sys
sys.path.append('..')
from file_and_folder_operations import read_data_list, myMakedirs, getSubFiles

def f_seg(case):
    predict_path = os.path.join(predict_dir, case.replace('.nii.gz', '_pred.nii.gz'))
    tr_seg = nib.load(predict_path)
    ras_seg = tr_seg.as_reoriented(io_orientation(tr_seg.affine))
    ras_seg_path = predict_path.replace('.nii.gz', '_reorientation.nii.gz')
    nib.save(ras_seg, ras_seg_path)

    tr_seg = sitk.ReadImage(ras_seg_path)
    spacing = np.array(tr_seg.GetSpacing())[[2, 1, 0]]
    size = np.array(tr_seg.GetSize())[[2, 1, 0]]
    tr_seg_array = sitk.GetArrayFromImage(tr_seg)

    # seg resample
    new_size = np.floor(((spacing / np.array(target_spacing)).astype(float) * size)).astype(int)
    seg_zoom = transform.resize(tr_seg_array.astype(float), new_size, 0, mode="edge", clip=True,
                                anti_aliasing=False).astype(np.uint8)

    np.savez_compressed(os.path.join(resample_dir, case.replace('.nii.gz', '.npz')), seg=seg_zoom)
    os.remove(ras_seg_path)
    print(case, size, new_size, spacing, target_spacing)

if __name__ == '__main__':

    '''resample mask for stage3'''
    target_spacing = [1., 0.6, 0.6]
    predict_dir = '/data/zym/experiment/noisy/stage2/update_mask'

    case_ls = os.listdir(predict_dir)
    case_ls = [i.replace('_pred.nii.gz', '.nii.gz') for i in case_ls]

    resample_dir = '/data/zym/workspace/noisy/stage2/update_label_resample_1.0X0.6X0.6/'

    myMakedirs(resample_dir, overwrite=False)
    with Pool(16) as p:
        p.map(f_seg, case_ls)