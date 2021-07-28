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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
sys.path.append('../')
from data.voxel2ply import write_ply

# img format: [C, H, W]
def center_image(img):
    img = img.astype(np.float32)
    var = np.var(img, axis=(1, 2), keepdims=True)
    mean = np.mean(img, axis=(1, 2), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


class Target(Dataset):
    def __init__(self, args):
        self.args = args
        self.pix3d_subclass_dict = {"bed": 0, "bookcase": 1, "chair": 2, "desk": 3,
                               "misc": 4, "sofa": 5, "table": 6, "tool": 7, "wardrobe": 8}
        pixel3d_path = os.path.join(self.args.data_dir, 'pix3d.hdf5')
        if os.path.exists(pixel3d_path):
            self.pixel3d = h5py.File(pixel3d_path, 'r')
        else:
            raise Exception("pixel3d not exist")

        sn_test_txt = os.path.join(self.args.data_dir, "shapenet_test.txt")
        self.test_idx = []

        for class_name in self.pix3d_subclass_dict.keys():
            sample_num = len(self.pixel3d[class_name]['pixels'])
            for sample_idx in range(sample_num):
                self.test_idx.append([class_name, sample_idx])

    def __len__(self):
        return len(self.test_idx)

    def __getitem__(self, idx):
        class_name, r_img_id = self.test_idx[idx]

        # pixel3d img as real r_img 
        r_class_id = self.pix3d_subclass_dict[class_name]
        #r_img_id = random.randint(0, self.pixel3d[subclass_name]['pixels'].shape[0] - 1 )
        r_img = self.pixel3d[class_name]['pixels'][int(r_img_id)]
        r_img = center_image(r_img)
        # r_img = preprocess_img(r_img)

        r_voxel = self.pixel3d[class_name]['voxels'][int(r_img_id)]
        r_voxel = np.reshape(r_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        sample = {'r_img_id': r_img_id,
                  'r_img': r_img,
                  'r_class_id': r_class_id,
                  'r_voxel': r_voxel,
                  }
        return sample


class DATest(Dataset):
    def __init__(self, args):
        self.args = args
        self.pix3d_subclass_dict = {"bed": 0, "bookcase": 1, "chair": 2, "desk": 3,
                               "misc": 4, "sofa": 5, "table": 6, "tool": 7, "wardrobe": 8}
        self.shapenet_subclass_dict = {"02818832": "bed", "03001627": "chair",
                                  "04256520": "sofa", "04379243": "table"}
        shapenet_path = os.path.join(self.args.data_dir, 'shapenet_train.hdf5')
        if os.path.exists(shapenet_path):
            shapenet = h5py.File(shapenet_path, 'r')
        else:
            raise Exception("shapenet not exist")

        pixel3d_path = os.path.join(self.args.data_dir, 'pix3d.hdf5')
        if os.path.exists(pixel3d_path):
            pixel3d = h5py.File(pixel3d_path, 'r')
        else:
            raise Exception("pixel3d not exist")

        sn_test_txt = os.path.join(self.args.data_dir, "shapenet_test.txt")
        self.test_idx = []

        sample_idx = 0
        with open(sn_test_txt, "r") as f:
            for line in f.readlines():
                strlist = line.split('/')
                class_name = strlist[0]
                if self.shapenet_subclass_dict.get(class_name):
                    self.test_idx.append(sample_idx)
                sample_idx += 1

        self.shapenet_vxls = shapenet['voxels']
        self.shapenet_pixels = shapenet['pixels']
        self.p3d_vxls = {}
        self.p3d_pixels = {}
        for cls in self.pix3d_subclass_dict:
            self.p3d_vxls[cls] = pixel3d[cls]['voxels']
            self.p3d_pixels[cls] = pixel3d[cls]['pixels']


    def __len__(self):
        return len(self.test_idx)

    def __getitem__(self, idx):
        sample_id = self.test_idx[idx]

        # shapenet as source dataset
        s_voxel = self.shapenet_vxls[sample_id]
        # 每组 24 张图片，从 24 张随机选取一张
        s_img_id = random.randint(0, self.shapenet_pixels[sample_id].shape[0] - 1)
        s_img = self.shapenet_pixels[sample_id][s_img_id]
        # s_img = preprocess_img(s_img)
        s_img = np.expand_dims(s_img, 0)
        s_img = center_image(s_img)

        # reshape to NCHW
        s_voxel = np.reshape(s_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])

        # pixel3d img as real r_img （randomly choose）
        p3d_subclass_list = [n for n in self.pix3d_subclass_dict.keys()]
        subclass_name = random.choice(p3d_subclass_list)
        subclass_id = self.pix3d_subclass_dict[subclass_name]
        r_img_id = random.randint(0, self.p3d_vxls[subclass_name].shape[0] - 1 )
        r_img = self.p3d_vxls[subclass_name][int(r_img_id)]
        r_img = center_image(r_img)
        # r_img = preprocess_img(r_img)

        r_voxel = self.p3d_vxls[subclass_name][int(r_img_id)]
        r_voxel = np.reshape(r_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])
        
        sample = {'s_img': s_img, 
                  's_voxel': s_voxel, 
                  'r_img': r_img,
                  'r_img_id': r_img_id,
                  'r_class_id': subclass_id,
                  'r_voxel': r_voxel,
                  }
        return sample


class DATrain(Dataset):
    def __init__(self,args):
        self.pseudo_label = None
        self.args = args

        pix3d_subclass_dict = {"bed": 0, "bookcase": 1, "chair": 2, "desk": 3,
                               "misc": 4, "sofa": 5, "table": 6, "tool": 7, "wardrobe": 8}
        # shapenet_subclass_dict = {"02818832": "bed", "03001627": "chair",
        #                           "04256520": "sofa", "table": "04379243"}
        shapenet_subclass_dict = None
        shapenet_path = os.path.join(args.data_dir, 'shapenet_train.hdf5')

        if os.path.exists(shapenet_path):
            shapenet = h5py.File(shapenet_path, 'r')
        else:
            raise Exception("shapenet not exist")

        pixel3d_path = os.path.join(args.data_dir, 'pix3d.hdf5')
        if os.path.exists(pixel3d_path):
            pixel3d = h5py.File(pixel3d_path, 'r')
        else:
            raise Exception("pixel3d not exist")

        sn_train_txt = os.path.join(args.data_dir, "shapenet_train.txt")
        train_idx = []

        sample_idx = 0
        with open(sn_train_txt, "r") as f:
            for line in f.readlines():
                strlist = line.split('/')
                class_name = strlist[0]
                if shapenet_subclass_dict is None:
                    train_idx.append(sample_idx)
                elif shapenet_subclass_dict.get(class_name):
                    train_idx.append(sample_idx)

                sample_idx += 1

        self.train_idx = train_idx
        self.shapenet_vxls = shapenet['voxels']
        self.shapenet_pixels = shapenet['pixels']
        self.p3d_vxls = {}
        self.p3d_pixels = {}
        for cls in pix3d_subclass_dict:
            self.p3d_vxls[cls] = pixel3d[cls]['voxels']
            self.p3d_pixels[cls] = pixel3d[cls]['pixels']

        self.pix3d_subclass_dict = pix3d_subclass_dict

    def __len__(self):
        return len(self.train_idx)

    def __getitem__(self, idx):
        sample_id = self.train_idx[idx]
        # shapenet as source dataset
        s_voxel = self.shapenet_vxls[sample_id]
        # 每组 24 张图片，从 24 张随机选取一张
        s_img_id = random.randint(0, self.shapenet_pixels[sample_id].shape[0] - 1)
        s_img = self.shapenet_pixels[sample_id][s_img_id]
        s_img = np.expand_dims(np.array(s_img), 0)
        #s_img = center_image(s_img)
        # reshape to NCHW
        s_voxel = np.reshape(s_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])

        # write_ply("%08d.ply" % sample_id, np.squeeze(s_voxel))

        # pixel3d img as real r_img （randomly choose）
        p3d_subclass_list = [n for n in self.pix3d_subclass_dict.keys()]
        subclass_name = random.choice(p3d_subclass_list)
        subclass_id = self.pix3d_subclass_dict[subclass_name]
        r_img_id = random.randint(0, self.p3d_pixels[subclass_name].shape[0] - 1 )
        r_img = self.p3d_pixels[subclass_name][int(r_img_id)]
        r_img = center_image(r_img)
        
        r_voxel = self.p3d_vxls[subclass_name][int(r_img_id)]
        r_voxel = np.reshape(r_voxel, [1, self.args.vox_size, self.args.vox_size, self.args.vox_size])

        #if self.pseudo_label and 'image_info' in self.pseudo_label.keys():
        #    reloc_id = self.pseudo_label['image_info'].index(subclass_id*1e5+r_img_id)
        #    ps_voxel = np.expand_dims(np.array(self.pseudo_label['pseudo_label'][reloc_id,::], 0))
        #else:
        #    ps_voxel = np.ones((1, self.args.vox_size, self.args.vox_size, self.args.vox_size))

        sample = {'s_img': s_img, 
                  's_voxel': s_voxel, 
                  'r_img': r_img,
                  'r_img_id': r_img_id,
                  'r_class_id': subclass_id,
                  'r_voxel': r_voxel,
                  }

        return sample


def load_data(args):
    # im_transforms = transforms.Compose([
    #                     transforms.RandomResizedCrop(256),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ToTensor()])
    train_loader = DataLoader(DATrain(args), batch_size=args.batch_size,
                            num_workers=16, pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(DATest(args), batch_size=1,
                                num_workers=16, pin_memory=True, shuffle=False)

    return train_loader, valid_loader


if __name__ == '__main__':
    def ls_dataset(name, node):
        print(node)

    h5py_file_path = '/media/qy/387E21207E20D900/DA3D/pix3d/pixel3d.hdf5'
    f = h5py.File(h5py_file_path, 'r')
    f.visititems(ls_dataset)



















