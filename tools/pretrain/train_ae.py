# -*- coding: utf-8 -*-
# voxel resolution: 32

import os
# import sys
# sys.path.append('..')
from model.backbone.ae_3d import *
import numpy as np
import h5py
import time
import argparse
import csv
from utils import evaluate_iou
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=500, type=int, help="Epoch to train [200]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="trained", help="Directory name to save the checkpoints [trained]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="/media/qy/387E21207E20D900/DA3D/dbs/", help="Root directory of dataset [dbs]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--train", action="store_true", dest="train", default=True, help="True for training, False for testing [False]")
parser.add_argument("--vox_size", action="store", dest="vox_size", default=32, type=int, help="Voxel resolution for training [32]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=32, type=int, help="batch_size for training [32]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
args = parser.parse_args()


def train(vxl_ae, train_loader, valid_loader, optimizer, device):

    start_time = time.time()
    criterion = nn.BCELoss()
    with open(os.path.join(args.checkpoint_dir, 'train_valid.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'valid_loss', 'valid_iou'])

    best_epoch = 0
    best_score = 0
    for epoch in range(0, args.epoch):
        vxl_ae.train()
        train_loss = 0
        train_num = 0
        for i, batch_voxels in enumerate(train_loader):
            batch_voxels = batch_voxels.float().to(device)

            vxl_ae.zero_grad()
            net_out = vxl_ae(batch_voxels)
            errSP = torch.mean(torch.abs(net_out-batch_voxels))
            errSP.backward()
            optimizer.step()

            train_loss += errSP.item()
            train_num += 1

        print(" Epoch(train): [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (
            epoch, args.epoch, time.time() - start_time, train_loss / train_num))

        with torch.no_grad():
            # valid model
            vxl_ae.eval()
            valid_loss = 0
            valid_num = 0
            valid_mean_iou = 0
            for i, batch_voxels in enumerate(valid_loader):
                batch_voxels = batch_voxels.float().to(device)
                net_out = vxl_ae(batch_voxels)
                # errSP = torch.mean(torch.abs(net_out-batch_voxels))
                errSP = criterion(net_out, batch_voxels)
                eval_ans = evaluate_iou(net_out.cpu().numpy(), batch_voxels.cpu().numpy(), 0.5)
                if eval_ans[1] > 0:
                    valid_mean_iou += np.mean(eval_ans[0] / eval_ans[1])

                valid_loss += errSP.item()
                valid_num += 1

            if valid_mean_iou > best_score:
                best_score = valid_mean_iou
                best_epoch = epoch
                save_dir = os.path.join(args.checkpoint_dir, "vxl_ae_best.pth")
                # save checkpoint
                torch.save(vxl_ae.state_dict(), save_dir)

            print(" Epoch(valid): [%2d/%2d] time: %4.4f, loss_sp: %.6f, iou: %.6f" % (
                epoch, args.epoch, time.time() - start_time, valid_loss / valid_num , valid_mean_iou / valid_num))

            with open(os.path.join(args.checkpoint_dir, 'train_valid.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([train_loss / train_num, valid_loss / valid_num, valid_mean_iou / valid_num])

    print('best epoch', best_epoch)
    # if not os.path.exists(args.checkpoint_dir):
    #     os.makedirs(args.checkpoint_dir)
    # save_dir = os.path.join(args.checkpoint_dir, "vxl_ae" + "-" + str(epoch) + ".pth")
    # # save checkpoint
    # torch.save(vxl_ae.state_dict(), save_dir)


if __name__ == '__main__':
    args.data_dir = '/home/qianyu.fqy/data/3D' #要修改为自己文件的地址
    # shapenet 读取方式
    train_data_path = os.path.join(args.data_dir, 'shapenet_train.hdf5')
    test_data_path = os.path.join(args.data_dir, 'shapenet_test.hdf5')
    if os.path.exists(train_data_path):
        train_data_dict = h5py.File(train_data_path, 'r')
        train_data = train_data_dict['voxels'][:]
        # reshape to NCHW
        train_data = np.reshape(train_data, [-1, 1, args.vox_size, args.vox_size, args.vox_size])
        train_data_dict.close()
        test_data_dict = h5py.File(test_data_path, 'r')
        test_data = test_data_dict['voxels'][:]
        # reshape to NCHW
        test_data = np.reshape(test_data, [-1, 1, args.vox_size, args.vox_size, args.vox_size])
        test_data_dict.close()

    else:
        print("error: cannot load " + train_data_path)
        exit(0)

    # pixel3d 读取方式
    # train_data_path = os.path.join(args.data_dir, 'pixel3d.hdf5')  # pixel3d
    # if os.path.exists(train_data_path):
    #     data_dict = h5py.File(train_data_path, 'r')
    #     # train_data = data_dict['/chair/voxles'][:]
    #     train_data = data_dict['chair']['voxels'][:]
    #     # reshape to NCHW
    #     train_data = np.reshape(train_data, [-1, 1, args.vox_size, args.vox_size, args.vox_size])
    # else:
    #     print("error: cannot load " + train_data_path)
    #     exit(0)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              num_workers=args.batch_size, pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(test_data, batch_size=args.batch_size,
                              num_workers=args.batch_size, pin_memory=True, shuffle=False)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    # build model
    vxl_ae = VoxelAE(32, 256).to(device)
    optimizer = torch.optim.Adam(vxl_ae.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    if args.train:
        train(vxl_ae, train_loader, valid_loader, optimizer, device)