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

# from utils import rescale_pix3d_voxel
# from sklearn.metrics.pairwise import cosine_similarity
use_transform = True

shapenet_train_filename = 'shapenet_train.hdf5'
shapenet_test_filename = 'shapenet_test.hdf5'

shapenet_subclass_dict = {"02818832": "bed", "03001627": "chair",
                          "04256520": "sofa", "04379243": "table",
                          "02691156": "aeroplane", "02958343": "car",
                          "03211117": "tvmonitor"}


pixel3d_filename = 'pix3d_rescale_chair.hdf5'  # 'pix3d.hdf5'

pix3d_subclass_dict = {"bed": 0, "bookcase": 1, "chair": 2, 
                       "desk": 3, "misc": 4, "sofa": 5, 
                       "table": 6, "tool": 7, "wardrobe": 8}


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
        self.src_data_cates = []
        with open(data_info, "r") as f:
            for line in f.readlines():
                class_id = str(line).split('/')[0]
                
                if class_id in shapenet_subclass_dict.keys():
                    sample_class = shapenet_subclass_dict[class_id]
                else:
                    sample_class = 'other'

                self.src_data_cates.append(sample_class)

        print('Source sample number: %d' % len(self.src_data_cates))

        # load chair images from pix3d as target
        file_path = os.path.join(args.data_dir, pixel3d_filename)
        self.pixel3d = h5py.File(file_path, 'r')['chair']
        self.trg_data_cates = ['chair']

        

        print('Target sample number: %d' % len(self.src_data_cates))

    def __len__(self):
        return len(self.src_data_cates)

    def __getitem__(self, idx):

        # source data
        s_img_id = random.randint(0, self.shapenet['pixels'][idx].shape[0] - 1)
        s_img = self.shapenet['pixels'][idx][s_img_id].astype(np.float32)
        s_class = self.src_data_cates[idx]
        s_voxel = self.shapenet['voxels'][idx].astype(np.float32)
        s_voxel = np.reshape(s_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        # target image
        t_img_id = random.randint(0, len(self.pixel3d) - 1)
        t_img_name = list(self.pixel3d.keys())[t_img_id]
        t_img = self.pixel3d[t_img_name]['image'][()].astype(np.float32)
        t_class = self.trg_data_cates[0]
        
        sim_label = 1 if s_class==t_class else 0

        # preprocess
        if use_transform:
            s_img = np.reshape(s_img, (224, 224))
            s_img = reshape_img(s_img)
            t_img = np.asarray(t_img)
        else:
            s_img = preprocess_s_img(s_img)
            t_img = preprocess_r_img(t_img)

        return {'s_img': s_img, 
                's_voxel': s_voxel, 
                't_img': t_img,
                'sim_label': torch.Tensor([sim_label])
                }


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
        t_img_id = self.test_idx[idx]
        t_img = self.pixel3d['pixels'][t_img_id].astype(np.float32)
        
        t_voxel = self.pixel3d['voxels'][t_img_id].astype(np.float32)
        t_voxel = np.reshape(t_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        t_img = t_img.transpose(1,2,0)
        t_img = preprocess_r_img(t_img)

        sample = {'t_img': t_img,
                  't_voxel': t_voxel
                  }
        return sample
