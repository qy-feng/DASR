import os
import numpy as np
import time
import argparse
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim.lr_scheduler import StepLR, ExponentialLR
# from warmup_scheduler import GradualWarmupScheduler
# import pytorch_warmup as warmup
from utils import *


def test(mdl, test_loader, device):
    mdl.eval()
    test_loss = 0
    test_score = 0
    test_num = 0
    criterion = nn.BCELoss()
    # eval_metric = 'cd'
    eval_metric = 'iou'
    move = True
    # move = False

    # results = []
    nan_num = 0 
    for i, batch_data in tqdm(enumerate(test_loader)):
        r_img = batch_data['r_img'].cuda()
        r_voxel = batch_data['r_voxel'].cuda()

        r_img_enc = mdl.f(r_img)
        r_img_rec = mdl.decoder(r_img_enc)

        errR = criterion(r_img_rec, r_voxel)
        test_loss += errR.item()
        
        # if move:
        #     r_img_rec_cb = []
        #     r_voxel_cb = []
        #     for b in range(r_img.shape[0]):
        #         r_img_rec_cb.append(move_center(r_img_rec[0,::]))
        #         r_voxel_cb.append(move_center(r_voxel[0,::]))
        #     r_img_rec_cb = torch.cat(r_img_rec_cb)
        #     r_voxel_cb = torch.cat(r_voxel_cb)

        # if i == 0:
        #     write_ply('pascal_%d.ply'%i, r_img_rec)
        #     write_ply('pascal_%d_gt.ply'%i, r_voxel)
        if move:
            r_img_rec = move_center(r_img_rec.squeeze())
            r_voxel = move_center(r_voxel.squeeze())
            
        if eval_metric == 'iou':
            r_img_rec = r_img_rec.detach().cpu().numpy()
            r_voxel = r_voxel.detach().cpu().numpy()
            intersection, union = evaluate_iou(r_img_rec, r_voxel, 0.5)
            if union > 0:
                test_score += np.mean(intersection / union)

        else:
            score = evaluate_cd(r_img_rec, r_voxel)
            if score!=score:
                test_score += 1
                nan_num += 1
            else:
                test_score += score.detach().cpu().item()
        test_num += 1
        torch.cuda.empty_cache()
    # print('None', nan_num)
    return test_loss / test_num, test_score / test_num


def move_center(voxel):
    voxel = voxel.squeeze() #.detach().cpu()
    curr_points = torch.nonzero(voxel >= 0.5)
    if len(curr_points) == 0:
        return voxel

    curr_center = torch.sum(curr_points, axis=0).float() / len(curr_points)
    target_center = torch.Tensor([15.5, 15.5, 15.5]).cuda()
    # print(curr_center.shape)
    mv_xyz = torch.round(target_center - curr_center).long()
    # print(curr_center, mv_xyz)
    mx, my, mz = mv_xyz[0], mv_xyz[1], mv_xyz[2]
    if mx >= 32 or mx < -32:
        return voxel
    if my >= 32 or my < -32:
        return voxel
    if mz >= 32 or mz < -32:
        return voxel
    # print('calibrate')
    voxel_ = torch.zeros_like(voxel).cuda()
    if mx >= 0:
        voxel_[mx:, :, :] = voxel[:32-mx, :, :]
    else:
        voxel_[:32+mx, :, :] = voxel[-mx:, :, :]
    voxel = voxel_.clone()
    voxel_ = torch.zeros_like(voxel).cuda()
    if my >= 0:
        voxel_[:, my:, :] = voxel[:, :32-my, :]
    else:
        voxel_[:, :32+my, :] = voxel[:, -my:, :]
    voxel = voxel_.clone()
    voxel_ = torch.zeros_like(voxel).cuda()
    if mz >= 0:
        voxel_[:, :, mz:] = voxel[:, :, :32-mz]
    else:
        voxel_[:, :, :32+mz] = voxel[:, :, -mz:]

    return voxel_
