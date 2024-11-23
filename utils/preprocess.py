import SimpleITK as sitk
import numpy as np
import os

# for my computer
# scans_path = r'E:\Chaokai_Tang\university\science-study\medical_image_segmentation\code\mycode\dataset\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\ct_scans'
# masks_path = r'E:\Chaokai_Tang\university\science-study\medical_image_segmentation\code\mycode\dataset\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\masks'

# scans_path_2d = r'E:\Chaokai_Tang\university\science-study\medical_image_segmentation\code\mycode\dataset\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\ct_2d'
# masks_path_2d = r'E:\Chaokai_Tang\university\science-study\medical_image_segmentation\code\mycode\dataset\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\mask_2d'


# for kaggle notebook
scans_path = '/kaggle/input/ct-zip/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1/ct_scans'
masks_path = '/kaggle/input/ct-zip/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1/masks'
scans_path_2d = '/kaggle/working/SAMIHS/dataset/ct_2d'
masks_path_2d = '/kaggle/working/SAMIHS/dataset/mask_2d'

ground_truths = os.listdir(masks_path)

if not os.path.exists(scans_path_2d):
    os.makedirs(scans_path_2d)

if not os.path.exists(masks_path_2d):
    os.makedirs(masks_path_2d)


for pa in os.listdir(scans_path):
    scan_img = sitk.ReadImage(os.path.join(scans_path, pa))
    assert pa in ground_truths
    mask_img = sitk.ReadImage(os.path.join(masks_path, pa))
    scan_arr = sitk.GetArrayFromImage(scan_img)
    label_arr = sitk.GetArrayFromImage(mask_img)
    for i in range(scan_arr.shape[0]):
        slice_name = 'BCIHM_' + pa.split('.')[0] + '_' + str(i).zfill(3) + '.npy'
        mask_name = 'BCIHM_' + pa.split('.')[0] + '_' + str(i).zfill(3) + '_seg' + '.npy'
        np.save(os.path.join(scans_path_2d, slice_name), scan_arr[i])
        np.save(os.path.join(masks_path_2d, mask_name), label_arr[i])
    print(pa, np.sum(scan_arr))