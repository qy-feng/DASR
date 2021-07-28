# -*- coding: utf-8 -*-
# voxel resolution: 32
import os
import sys
sys.path.append('../')
from data.loader import load_data

import numpy as np
import time
import argparse
import csv
from utils import evaluate_iou
from model.naive_vxl_ae import VoxelNaiveAE
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=500, type=int, help="Epoch to train [200]")
parser.add_argument("--code_len", action="store", dest="code_len", default=256, type=int, help="len of laten code")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00001, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [trained]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="/media/qy/387E21207E20D900/DA3D/dbs/", help="Root directory of dataset [dbs]")
parser.add_argument("--train", action="store_true", dest="train", default=True, help="True for training, False for testing [False]")
parser.add_argument("--vox_size", action="store", dest="vox_size", default=32, type=int, help="Voxel resolution for training [32]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=4, type=int, help="batch_size for training [32]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=32, type=int, help="In testing, output shapes [start:end]")
args = parser.parse_args()


def weighted_loss(pre_vxls, gt_vxls):
    abs_diff = torch.abs((gt_vxls - pre_vxls))
    mask = (gt_vxls > 0).float()
    mask_num = torch.sum(mask)
    w = 0.5
    loss1 = torch.sum(abs_diff * mask) / mask_num
    loss2 = torch.sum(abs_diff * (1.0 - mask)) / (gt_vxls.numel() - mask_num)
    loss = w * loss1 + (1.0 - w) * loss2
    return loss

def train(vxl_ae, train_loader, valid_loader, optimizer, device):

    start_time = time.time()

    with open(os.path.join(args.checkpoint_dir, 'naive_ae_train_valid.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'valid_loss', 'valid_iou'])

    for epoch in range(0, args.epoch):
        vxl_ae.train()
        train_loss = 0
        train_num = 0
        for i, batch_data in enumerate(train_loader):
            batch_voxels = batch_data['s_voxel'].float().to(device)
            gt_vxls = torch.clone(batch_voxels)

            vxl_ae.zero_grad()
            net_out = vxl_ae(batch_voxels)
            # recon_loss = torch.mean(torch.abs(net_out - batch_voxels))
            recon_loss = torch.nn.functional.binary_cross_entropy(net_out, gt_vxls)
            # recon_loss = weighted_loss(net_out, batch_voxels)
            recon_loss.backward()
            optimizer.step()

            train_loss += recon_loss.item()
            train_num += 1

        print(" Epoch(train): [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (
            epoch, args.epoch, time.time() - start_time, train_loss / train_num))

        if epoch % 50 == 49:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            save_dir = os.path.join(args.checkpoint_dir, "vxl_naive_ae" + "-" + str(epoch) + ".pth")
            # save checkpoint
            torch.save(vxl_ae.state_dict(), save_dir)

        with torch.no_grad():
            # valid model
            vxl_ae.eval()
            valid_loss = 0
            valid_num = 0
            valid_mean_iou = 0
            for i, batch_data in enumerate(valid_loader):
                batch_voxels = batch_data['s_voxel'].float().to(device)
                gt_vxls = torch.clone(batch_voxels)

                net_out = vxl_ae(batch_voxels)
                # recon_loss = torch.mean(torch.abs(net_out - batch_voxels))
                # recon_loss = weighted_loss(net_out, batch_voxels)
                recon_loss = torch.nn.functional.binary_cross_entropy(net_out, gt_vxls)

                intersection, union = evaluate_iou(net_out.cpu().numpy(), gt_vxls.cpu().numpy(), 0.5)
                if union == 0:
                    valid_num -= 1
                else:
                    valid_mean_iou += np.mean(intersection / union)

                valid_loss += recon_loss.item()
                valid_num += 1

            print(" Epoch(valid): [%2d/%2d] time: %4.4f, loss_sp: %.6f, vnum: %d iou: %.6f" % (
                epoch, args.epoch, time.time() - start_time, valid_loss / valid_num , valid_num, valid_mean_iou / valid_num))

            with open(os.path.join(args.checkpoint_dir, 'naive_ae_train_valid.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([train_loss / train_num, valid_loss / valid_num, valid_mean_iou / valid_num])


    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    save_dir = os.path.join(args.checkpoint_dir, "vxl_naive_ae" + "-" + str(epoch) + ".pth")
    # save checkpoint
    torch.save(vxl_ae.state_dict(), save_dir)


if __name__ == '__main__':
    train_loader, valid_loader = load_data(args)

    print("train sample #: %d" % len(train_loader))
    print("valid sample #: %d" % len(valid_loader))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    # build model
    vxl_naive_ae = VoxelNaiveAE(32, args.code_len).to(device)
    optimizer = torch.optim.Adam(vxl_naive_ae.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    if args.train:
        train(vxl_naive_ae, train_loader, valid_loader, optimizer, device)