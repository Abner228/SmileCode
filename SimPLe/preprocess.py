import numpy as np
from file_and_folder_operations import getSubFiles, myMakedirs
from skimage import transform
import os
import SimpleITK as sitk
import nibabel as nib
from nibabel import io_orientation
from multiprocessing import Pool

def f_img(tr_img_path):
    # reorient to ras
    tr_img = sitk.ReadImage(tr_img_path)
    direction = np.array(tr_img.GetDirection())
    origin = np.array(tr_img.GetOrigin())
    spacing = np.array(tr_img.GetSpacing())[[2, 1, 0]]
    size = np.array(tr_img.GetSize())[[2, 1, 0]]

    tr_img = nib.load(tr_img_path)
    origin_affine = tr_img.affine
    ras_img = tr_img.as_reoriented(io_orientation(tr_img.affine))
    ras_img_path = tr_img_path.replace('raw_data', 'resample_1.0X0.6X0.6').replace('.nii.gz', '_reorientation.nii.gz')
    nib.save(ras_img, ras_img_path)

    tr_img = sitk.ReadImage(ras_img_path)
    tr_img_array = sitk.GetArrayFromImage(tr_img)

    # img resample
    new_size = np.floor(((spacing / np.array(target_spacing)).astype(float) * size)).astype(int)
    img_zoom = transform.resize(tr_img_array, new_size, order=3, preserve_range=True)

    np.savez_compressed(tr_img_path.replace('raw_data', 'resample_1.0X0.6X0.6').replace('.nii.gz', '.npz'),
                        data=img_zoom,
                        origin_spacing=spacing,
                        origin_size=size,
                        origin_direction=direction,
                        origin_affine=origin_affine,
                        origin=origin)
    os.remove(ras_img_path)
    print(tr_img_path, size, new_size, spacing, target_spacing)

def f_seg(tr_seg_path):
    # reorient to ras
    tr_seg = nib.load(tr_seg_path)
    ras_seg = tr_seg.as_reoriented(io_orientation(tr_seg.affine))
    ras_seg_path = tr_seg_path.replace('noisy_label', 'noisy_label_resample_1.0X0.6X0.6').replace('.nii.gz', '_reorientation.nii.gz')
    nib.save(ras_seg, ras_seg_path)

    tr_seg = sitk.ReadImage(ras_seg_path)
    spacing = np.array(tr_seg.GetSpacing())[[2, 1, 0]]
    size = np.array(tr_seg.GetSize())[[2, 1, 0]]
    tr_seg_array = sitk.GetArrayFromImage(tr_seg)

    # seg resample
    new_size = np.floor(((spacing / np.array(target_spacing)).astype(float) * size)).astype(int)
    seg_zoom = transform.resize(tr_seg_array.astype(float), new_size, 0, mode="edge", clip=True,
                                anti_aliasing=False).astype(np.uint8)

    np.savez_compressed(tr_seg_path.replace('noisy_label', 'noisy_label_resample_1.0X0.6X0.6').replace('.nii.gz', '.npz'),
                        seg=seg_zoom)
    os.remove(ras_seg_path)
    print(tr_seg_path, size, new_size, spacing, target_spacing)

def f_check(seg_path):
    print(seg_path, end=' ')
    seg = np.load(seg_path)
    seg_data = seg['seg']
    case = seg_path.split('/')[-1]
    img = np.load(os.path.join('/data/zym/YN/DCE/resample_1.0X0.6X0.6/DCE_C1', case))
    img_data = img['data']

    print(img_data.shape, seg_data.shape)
    assert img_data.shape == seg_data.shape

if __name__ == '__main__':

    target_spacing = [1., 0.6, 0.6]

    '''resample dce c1'''
    img_dir = '/data/zym/YN/DCE/raw_data/DCE_C1'
    img_path_list = getSubFiles(img_dir, '.nii.gz')
    resample_dir = img_dir.replace('raw_data', 'resample_1.0X0.6X0.6')
    myMakedirs(resample_dir, overwrite=False)
    with Pool(16) as p:
        p.map(f_img, img_path_list)

    '''resample mask'''
    seg_dir = '/data/zym/workspace/noisy/noisy_label'
    seg_path_list = getSubFiles(seg_dir, '.nii.gz')
    resample_dir = seg_dir.replace('noisy_label', 'noisy_label_resample_1.0X0.6X0.6')
    myMakedirs(resample_dir, overwrite=False)
    with Pool(16) as p:
        p.map(f_seg, seg_path_list)

    '''check'''
    noisy_label_path_list = getSubFiles('/data/zym/workspace/noisy/noisy_label_resample_1.0X0.6X0.6', '.npz')
    with Pool(16) as p:
        p.map(f_check, noisy_label_path_list)
