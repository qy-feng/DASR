# -*- coding: utf-8 -*-

# 先对转换之后的数据的格式进行统一：
# 使用 hdf5 的格式进行存储，有多少类别就有多少个 group，每个 group 包含
# pixels, shape 为 [num_instance, num_img, rows, cols]，灰度图形式存储
# voxels, shape 为 [num_instance, rows, cols, depths, 1]，每个 instance 对应一个 voxel

import json
from scipy.io import loadmat
import os
import numpy as np
import cv2
import h5py
from PIL import Image, ImageOps
from tqdm import tqdm
import sys
sys.path.append('../')
from utils import voxel_downsampling

DST_FILE = 'color_pixel3d.hdf5'


def resize_padding(im, desired_size):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size), (255, 255, 255))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


raw_db_root = '/media/qy/387E21207E20D900/DA3D/pix3d'
raw_db = json.load(open(os.path.join(raw_db_root, 'pix3d.json')))
dst_db_path = os.path.join(raw_db_root, DST_FILE)
num_sample = len(raw_db)
samples = {}

for idx in tqdm(np.arange(num_sample)):
    sample = raw_db[idx]
    raw_vxl = loadmat(os.path.join(raw_db_root, sample['voxel']))
    raw_vxl = raw_vxl['voxel']
    category = sample['category']

    if samples.get(category) is None:
        samples[category] = [[], []]

    raw_im = Image.open(os.path.join(raw_db_root, sample['img']))
    raw_im = raw_im.convert('RGB')
    raw_im = np.array(raw_im)
    mask = Image.open(os.path.join(raw_db_root, sample['mask']))
    mask = np.array(mask)
    x, y, w, h = cv2.boundingRect(mask)
    dst_img = raw_im[y:y + h, x:x + w, :]
    dst_img = Image.fromarray(dst_img)
    dst_img = resize_padding(dst_img, 224)
    # dst_img = np.array(ImageOps.grayscale(dst_img))
    dst_img = np.array(dst_img)
    dst_img = np.moveaxis(dst_img, -1, 0)
    # print("dst img shape: ", dst_img.shape)

    dst_vxl = voxel_downsampling(raw_vxl, 0.25)

    dst_vxl = np.expand_dims(dst_vxl, axis=-1)
    # dst_img = np.expand_dims(dst_img, axis=0)

    samples[category][0].append(dst_img)
    samples[category][1].append(dst_vxl)

f = h5py.File(dst_db_path, 'w')
for _category, data in samples.items():
    g = f.create_group(_category)

    pixels = np.stack(data[0], axis=0)
    voxels = np.stack(data[1], axis=0)

    print('\n\n', _category)
    print(pixels.shape)
    print(voxels.shape)

    g.create_dataset('pixels', data=pixels)
    g.create_dataset('voxels', data=voxels)

f.close()


def ls_dataset(name, node):
    print(node)


h5py_file_path = '/media/qy/387E21207E20D900/DA3D/pix3d/' + DST_FILE
f = h5py.File(h5py_file_path, 'r')
f.visititems(ls_dataset)
