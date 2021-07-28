# -*- coding: utf-8 -*-
import h5py
import os
import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import voxel_downsampling
from tqdm import tqdm


def ls_dataset(name, node):
    print(node)


def convert_train_data():
    h5py_file_path = '/media/qy/387E21207E20D900/DA3D/dbs/all_vox256_img_train.hdf5'
    f = h5py.File(h5py_file_path, 'r')
    pixels = f['pixels']
    voxels = f['voxels']

    raw_db_root = '/media/qy/387E21207E20D900/DA3D/dbs'
    dst_db_path = os.path.join(raw_db_root, 'shapenet_train.hdf5')

    offset_x = int((137 - 128) / 2)
    offset_y = int((137 - 128) / 2)
    # reshape to NCHW
    pixels = np.reshape(pixels[:, :, offset_y:offset_y + 128, offset_x:offset_x + 128], [-1, 24, 128, 128])
    voxels = np.squeeze(voxels, -1)

    dst_voxels = np.zeros((voxels.shape[0], 32, 32, 32), dtype=np.int8)
    dst_pixels = np.zeros((pixels.shape[0], pixels.shape[1], 224, 224), dtype=np.int8)

    for sid in tqdm(np.arange(len(pixels))):
        dst_voxels[sid] = voxel_downsampling(voxels[sid], 0.5)
        for iid in np.arange(len(pixels[sid])):
            cv2.resize(pixels[sid, iid], dsize=(224, 224), dst=dst_pixels[sid, iid], interpolation=cv2.INTER_LINEAR)

    dst_voxels = np.expand_dims(dst_voxels, -1)
    dst_f = h5py.File(dst_db_path, 'w')
    dst_f.create_dataset('pixels', data=dst_pixels)
    dst_f.create_dataset('voxels', data=dst_voxels)

    dst_f.visititems(ls_dataset)
    dst_f.close()


def convert_test_data():
    h5py_file_path = '/media/qy/387E21207E20D900/DA3D/dbs/all_vox256_img_test.hdf5'
    f = h5py.File(h5py_file_path, 'r')
    pixels = f['pixels']
    voxels = f['voxels']

    raw_db_root = '/media/qy/387E21207E20D900/DA3D/dbs'
    dst_db_path = os.path.join(raw_db_root, 'shapenet_test.hdf5')

    offset_x = int((137 - 128) / 2)
    offset_y = int((137 - 128)  / 2)
    # reshape to NCHW
    pixels = np.reshape(pixels[:, :, offset_y:offset_y + 128, offset_x:offset_x + 128], [-1, 24, 128, 128])
    voxels = np.squeeze(voxels, -1)

    dst_voxels = np.zeros((voxels.shape[0], 32, 32, 32), dtype=np.int8)
    dst_pixels = np.zeros((pixels.shape[0], pixels.shape[1], 224, 224), dtype=np.int8)

    for sid in tqdm(np.arange(len(pixels))):
        dst_voxels[sid] = voxel_downsampling(voxels[sid], 0.5)
        for iid in np.arange(len(pixels[sid])):
            cv2.resize(pixels[sid, iid], dsize=(224, 224), dst=dst_pixels[sid, iid],  interpolation=cv2.INTER_LINEAR)

    dst_voxels = np.expand_dims(dst_voxels, -1)
    dst_f = h5py.File(dst_db_path, 'w')
    dst_f.create_dataset('pixels', data=dst_pixels)
    dst_f.create_dataset('voxels', data=dst_voxels)

    dst_f.visititems(ls_dataset)
    dst_f.close()


if __name__ == '__main__':
    print('convert train data\n')
    convert_train_data()

    print('\nconvert test data\n')
    convert_test_data()