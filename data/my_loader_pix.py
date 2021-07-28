## Implement a data loader for 3D domain adaptation task
##
## Output (paired): 
## - synthetic 2d img
## - synthetic 3d voxel
## - real 2d img

import os
import sys
import numpy as np
import h5py
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import random
sys.path.append('../')
from .transforms import *
import pickle as pkl
from utils import rescale_pix3d_voxel
# from sklearn.metrics.pairwise import cosine_similarity
################# Tuning #################
use_subset = False
# use_subset = True

##########################################
use_transform = True

pixel3d_filename = 'pix3d_rescale_chair.hdf5'  # 'pix3d_2894_chair.hdf5'  
# 'color_pixel3d.hdf5'  # 'pix3d.hdf5'
shapenet_train_filename = 'shapenet_train.hdf5'
shapenet_test_filename = 'shapenet_test.hdf5'
pascal_image_filename = 'pascal3d_images.hdf5'
pascal_dir = 'chair_pascal'

shapenet_subclass_dict = {"02818832": "bed", "03001627": "chair",
                          "04256520": "sofa", "04379243": "table"}
pix3d_subclass_dict = {"bed": 0, "bookcase": 1, "chair": 2, 
                       "desk": 3, "misc": 4, "sofa": 5, 
                       "table": 6, "tool": 7, "wardrobe": 8}


def load_data(args):
    train_dataset = Train(args)
    test_dataset = Test(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        if use_transform:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, 
                                                pin_memory=True, drop_last=True,
                                                collate_fn=my_collate_fn, sampler=train_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, 
                                                pin_memory=True, shuffle=True, drop_last=True, 
                                                sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, 
                                                pin_memory=True, sampler=test_sampler)                                        
    else:
        if use_transform:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, 
                                                pin_memory=True, shuffle=True, drop_last=True,
                                                collate_fn=my_collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, 
                                                pin_memory=True, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, 
                                                pin_memory=True, shuffle=False)

    return train_loader, test_loader


class Train(Dataset):
    def __init__(self, args):
        self.args = args

        # load shapenet (source)
        shapenet_path = os.path.join(args.data_dir, shapenet_train_filename)
        if os.path.exists(shapenet_path):
            self.shapenet = h5py.File(shapenet_path, 'r')
        else:
            raise Exception("shapenet not exist")

        sn_train_txt = os.path.join(args.data_dir, "shapenet_train.txt")
        self.train_idx = []
        self.shn_chair_idx = []
        self.sim_labels = []
        sample_id = 0
        with open(sn_train_txt, "r") as f:
            for line in f.readlines():
                if str(line).split('/')[0] == "03001627": # chair
                    sim_label = 1
                    self.shn_chair_idx.append(sample_id)
                else:
                    sim_label = 0
                self.train_idx.append(sample_id)
                self.sim_labels.append(sim_label)
                sample_id += 1
        if use_subset:
            self.train_idx = self.train_idx[:len(self.train_idx)//100]
        print('train sample num: %d' % len(self.train_idx))

        # load pascal chair images
        # self.pascal_dir = os.path.join(args.data_dir, pascal_dir)
        # self.pascal_img_list = os.listdir(self.pascal_dir)
        pascal_img_path = os.path.join(args.data_dir, pascal_image_filename)
        self.pascal_imgs = h5py.File(pascal_img_path, 'r')['chair']

    def __len__(self):
        return len(self.train_idx)

    def __getitem__(self, idx):
        sample_id = self.train_idx[idx]

        # synthetic image + voxel
        s_img_id = random.randint(0, self.shapenet['pixels'][sample_id].shape[0] - 1)
        s_img = self.shapenet['pixels'][sample_id][s_img_id].astype(np.float32)

        s_voxel = self.shapenet['voxels'][sample_id].astype(np.float32)
        s_voxel = np.reshape(s_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        # natural image
        r_img_id = random.randint(0, len(self.pascal_imgs) - 1)
        # r_img = Image.open(os.path.join(self.pascal_dir, self.pascal_img_list[r_img_id]))
        r_img_name = list(self.pascal_imgs.keys())[r_img_id]
        r_img = self.pascal_imgs[r_img_name]['image'][()].astype(np.float32)
        r_size = r_img.shape
        
        # image preprocess
        if use_transform:
            s_img = np.reshape(s_img, (224, 224))
            s_img = reshape_img(s_img)
            
            r_img = np.asarray(r_img)
            # if r_img.size // r_size[0] // r_size[1] == 3:
            #     r_img = np.reshape(r_img, (r_size[0], r_size[1], 3))
            #     r_img = Image.fromarray(r_img.astype('uint8'))
            # else:
            #     r_img = reshape_img(np.reshape(r_img, r_size))
        else:
            s_img = preprocess_s_img(s_img)

            r_img = np.reshape(np.array(r_img), (3, r_size[1], r_size[2]))
            print('1', r_img.shape)
            r_img = r_img.transpose(1,2,0)
            print('2', r_img.shape)
            r_img = preprocess_r_img(r_img)
            print('3', r_img.shape)
        # print(type(r_img), r_img.size)

        sample = {'s_img': s_img, 
                  's_voxel': s_voxel, 
                  'r_img': r_img,
                  'sim_label': torch.Tensor([self.sim_labels[idx]])
                 }

        return sample


class Test(Dataset):
    def __init__(self, args):
        self.args = args
        
        pixel3d_path = os.path.join(args.data_dir, pixel3d_filename)
        if os.path.exists(pixel3d_path):
            self.pixel3d = h5py.File(pixel3d_path, 'r')
        else:
            raise Exception("pixel3d not exist")

        self.test_idx = []
        sample_num = len(self.pixel3d['pixels'])
        for sample_idx in range(sample_num):
            self.test_idx.append(sample_idx)

        if use_subset:
            self.test_idx = self.test_idx[:len(self.test_idx)//100]
        print('test sample num: %d' % len(self.test_idx))

    def __len__(self):
        return len(self.test_idx)

    def __getitem__(self, idx):
        r_img_id = self.test_idx[idx]
        r_img = self.pixel3d['pixels'][r_img_id].astype(np.float32)
        
        r_voxel = self.pixel3d['voxels'][r_img_id].astype(np.float32)
        r_voxel = np.reshape(r_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        r_img = r_img.transpose(1,2,0)
        r_img = preprocess_r_img(r_img)

        sample = {'r_img': r_img,
                  'r_voxel': r_voxel
                  }
        return sample
