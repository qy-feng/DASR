import os
import sys
import numpy as np
import h5py
import pickle as pkl
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import random
sys.path.append('/data/qianyu/3d/DA-vx')
from dataloader.transforms import *

use_transform = True

shapenet_train_filename = 'shapenet_train.hdf5'
shapenet_test_filename = 'shapenet_test.hdf5'

shapenet_subclass_dict = {"02818832": "bed", "03001627": "chair",
                          "04256520": "sofa", "04379243": "table",
                          "02691156": "aeroplane", "02958343": "car",
                          "03211117": "tvmonitor"}

imagenet_filename = 'imagenet_images.hdf5'
pascal_image_filename = 'pascal3d_images.hdf5'
# pascal_voxel_filename = 'pascal3d_voxels.hdf5'
pascal_voxel_filename = 'rescaled_pascal3d_voxles.hdf5'

pascal_subclass = ['aeroplane', 'car', 'chair', 'tvmonitor', 'diningtable']


class Train(Dataset):
    def __init__(self, args):
        self.args = args

        # load shapenet as source
        shapenet_path = os.path.join(args.data_dir, shapenet_train_filename)
        if os.path.exists(shapenet_path):
            self.shapenet = h5py.File(shapenet_path, 'r')
        else:
            raise Exception("shapenet not exist")

        data_info = os.path.join(args.data_dir, "shapenet_train.txt")
        self.src_data_cates = dict()
        with open(data_info, "r") as f:
            s_idx = 0
            for line in f.readlines():
                class_id = str(line).split('/')[0]
                
                if class_id in shapenet_subclass_dict.keys():
                    class_name = shapenet_subclass_dict[class_id]
                else:
                    class_name = 'other'
                if class_name not in self.src_data_cates.keys():
                    self.src_data_cates[class_name] = []
                
                self.src_data_cates[class_name].append(s_idx)
                s_idx += 1
        
        print('Source sample number: %d' % len(self.src_data_cates))

        # load imagenet images from pascal3d+ as target
        file_path = os.path.join(args.data_dir, imagenet_filename)
        self.imagenet = h5py.File(file_path, 'r')

        self.trg_sample_idx = []
        for cate in self.imagenet.keys():
            for t_idx in range(len(self.imagenet[cate])):
                self.trg_sample_idx.append((cate, t_idx))

        print('Target sample number: %d' % len(self.trg_sample_idx))

    def __len__(self):
        return len(self.trg_sample_idx)

    def sample_cate(self, cate):
        s_idxs = self.src_data_cates[cate]
        s_idx = random.choice(s_idxs)  # s_idxs[random.randint(0, len(s_idxs) - 1)]
        
        # random sample an image
        img_idxs = self.shapenet['pixels'][s_idx].keys()
        img_idx = random.choice(img_idxs)  # [random.randint(0, len(img_idxs) - 1)]
        img = self.shapenet['pixels'][s_idx][img_idx].astype(np.float32)

        voxel = self.shapenet['voxels'][s_idx].astype(np.float32)
        voxel = np.reshape(voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        return img, voxel

    def __getitem__(self, idx):
        # target sample
        trg_cate, trg_idx = self.trg_sample_idx[idx]
        t_img = self.imagenet[trg_cate][trg_idx]['image']

        # source 
        ## positive sample
        pos_cate = trg_cate
        s_img_pos, s_vxl_pos = self.sample_cate(pos_cate)
        # s_img_id = random.randint(0, self.shapenet['pixels'][idx].shape[0] - 1)
        # s_img = self.shapenet['pixels'][idx][s_img_id].astype(np.float32)
        # src_cate = self.src_data_cates[idx]
        # s_voxel = self.shapenet['voxels'][idx].astype(np.float32)
        # s_voxel = np.reshape(s_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        ## negative sample
        neg_cates = list(self.src_data_cates.keys()).pop(pos_cate)
        s_img_neg, s_vxl_neg = self.sample_cate(random.choice(neg_cates))

        # preprocess
        if use_transform:
            s_img_pos = reshape_img(s_img_pos)
            s_img_neg = reshape_img(s_img_neg)
            t_img = np.asarray(t_img).transpose(1,2,0)
            # t_img = Image.fromarray(t_img.astype('uint8'))
        else:
            s_img_pos = preprocess_s_img(s_img_pos)
            s_img_neg = reshape_img(s_img_neg)
            t_img = preprocess_r_img(t_img)

        return {'s_img_pos': s_img_pos, 
                's_img_neg': s_img_neg,
                's_vxl_pos': s_vxl_pos, 
                's_vxl_neg': s_vxl_neg,
                't_img': t_img}


class Test(Dataset):
    def __init__(self, args):
        self.args = args
        
        pascal_image_path = os.path.join(args.data_dir, pascal_image_filename)
        self.pascal_image = h5py.File(pascal_image_path, 'r')
        pascal_voxel_path = os.path.join(args.data_dir, pascal_voxel_filename)
        self.pascal_voxel = h5py.File(pascal_voxel_path, 'r')

        self.test_idx = []
        for t_cls in self.pascal_image.keys():
            print(t_cls)
            for t_img in self.pascal_image[t_cls].keys():
                t_vxl_id = self.pascal_image[t_cls][t_img]['vxl_idx'][()]
                self.test_idx.append((t_cls, t_img, t_vxl_id))

        print('test sample num: %d' % len(self.test_idx))

    def __len__(self):
        return len(self.test_idx)

    def __getitem__(self, idx):
        t_cls, t_img, t_vxl_id = self.test_idx[idx]
        t_img = self.pascal_image[t_cls][t_img]['image'].astype(np.float32)
        
        t_voxel = self.pascal_voxel[t_cls][t_vxl_id].astype(np.float32)
        t_voxel = np.reshape(t_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        t_img = np.asarray(t_img).transpose(1,2,0)
        t_img = preprocess_r_img(t_img)

        sample = {'t_img': t_img,
                  't_voxel': t_voxel
                  }
        return sample
