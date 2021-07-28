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

################# Tuning #################
use_subset = False
# use_subset = True

##########################################
use_transform = True

shapenet_train_filename = 'shapenet_train.hdf5'
shapenet_test_filename = 'shapenet_test.hdf5'
imagenet_filename = 'imagenet_images.hdf5'
pascal_image_filename = 'pascal3d_images.hdf5'
# pascal_voxel_filename = 'pascal3d_voxels.hdf5'
pascal_voxel_filename = 'rescaled_pascal3d_voxles.hdf5'


shapenet_subclass_dict = {"02818832": "bed", "03001627": "chair",
                          "04256520": "sofa", "04379243": "table",
                          "02691156": "aeroplane", "02958343": "car",
                          "03211117": "tvmonitor"}
pix3d_subclass_dict = {"bed": 0, "bookcase": 1, "chair": 2, 
                       "desk": 3, "misc": 4, "sofa": 5, 
                       "table": 6, "tool": 7, "wardrobe": 8}
pascal_subclass = ['aeroplane', 'car', 'chair', 'tvmonitor']

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
        sample_id = 0
        with open(sn_train_txt, "r") as f:
            for line in f.readlines():
                class_id = str(line).split('/')[0]
                # if str(line).split('/')[0] == "03001627": # chair
                #     self.shn_chair_idx.append(sample_id)
                if class_id in shapenet_subclass_dict.keys():
                    sample_class = shapenet_subclass_dict[class_id]
                else:
                    sample_class = 'other'
                self.train_idx.append((sample_id, sample_class))
                sample_id += 1
        if use_subset:
            self.train_idx = self.train_idx[:len(self.train_idx)//100]
        print('train sample num: %d' % len(self.train_idx))

        # load imagenet
        imagenet_path = os.path.join(args.data_dir, imagenet_filename)
        self.imagenet = h5py.File(imagenet_path, 'r')
        self.imagenet_class_img = dict()
        for r_cls in self.imagenet.keys():
            self.imagenet_class_img[r_cls] = list(self.imagenet[r_cls].keys())

    def __len__(self):
        return len(self.train_idx)

    def __getitem__(self, idx):
        sample_id, sample_class = self.train_idx[idx]

        # shapenet image + voxel
        s_img_id = random.randint(0, self.shapenet['pixels'][sample_id].shape[0] - 1)
        s_img = self.shapenet['pixels'][sample_id][s_img_id].astype(np.float32)

        s_voxel = self.shapenet['voxels'][sample_id].astype(np.float32)
        s_voxel = np.reshape(s_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        # imagenet image
        r_cls_id = random.randint(0, len(self.imagenet_class_img.keys()) - 1)
        r_class = list(self.imagenet_class_img.keys())[r_cls_id]
        r_img_id = random.randint(0, len(self.imagenet_class_img[r_class]) - 1)
        r_img_name = self.imagenet_class_img[r_class][r_img_id]
        r_img = self.imagenet[r_class][r_img_name]['image']
        r_size = r_img.shape
        # print(r_size)
        
        sim_label = 1 if sample_class==r_class else 0

        # image preprocess
        if use_transform:
            s_img = np.reshape(s_img, (224, 224))
            s_img = reshape_img(s_img)
            r_img = np.asarray(r_img).transpose(1,2,0)
            r_img = Image.fromarray(r_img.astype('uint8'))
        else:
            s_img = preprocess_s_img(s_img)
            r_img = preprocess_r_img(r_img)

        sample = {'s_img': s_img, 
                  's_voxel': s_voxel,
                  'sim_label': torch.Tensor([sim_label]),
                  'r_img': r_img
                 }

        return sample


class Test(Dataset):
    def __init__(self, args):
        self.args = args
        
        pascal_image_path = os.path.join(args.data_dir, pascal_image_filename)
        self.pascal_image = h5py.File(pascal_image_path, 'r')
        pascal_voxel_path = os.path.join(args.data_dir, pascal_voxel_filename)
        self.pascal_voxel = h5py.File(pascal_voxel_path, 'r')

        self.test_idx = []
        # for r_cls in [pascal_subclass[0]]:
        for r_cls in self.pascal_image.keys():
            print(r_cls)
            for r_img in self.pascal_image[r_cls].keys():
                r_vxl_id = self.pascal_image[r_cls][r_img]['vxl_idx'][()]
                self.test_idx.append((r_cls, r_img, r_vxl_id))

        if use_subset:
            self.test_idx = self.test_idx[:len(self.test_idx)//100]
        print('test sample num: %d' % len(self.test_idx))

    def __len__(self):
        return len(self.test_idx)

    def __getitem__(self, idx):
        r_cls, r_img, r_vxl_id = self.test_idx[idx]
        r_img = self.pascal_image[r_cls][r_img]['image']#.astype(np.float32)
        
        r_voxel = self.pascal_voxel[r_cls][r_vxl_id].astype(np.float32)
        r_voxel = np.reshape(r_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        r_img = np.asarray(r_img).transpose(1,2,0)
        r_img = preprocess_r_img(r_img)

        sample = {'r_img': r_img,
                  'r_voxel': r_voxel
                  }
        return sample
