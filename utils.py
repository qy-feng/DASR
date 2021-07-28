# -*- coding: utf-8 -*-
# reference: https://github.com/chrischoy/3D-R2N2/blob/7522834a6d155ac5c30e3119e3747f0967b299d2/lib/voxel.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler
from chamfer_distance.chamfer_distance_gpu import ChamferDistance


def evaluate_cd(preds, gts):
    chamfer_dist = ChamferDistance()
    batch_size = preds.shape[0]
    cd_score = 0
    for b in range(batch_size):
        pred = preds[b, ::]
        gt = gts[b, ::]
        
        # convert voxel to pc
        # pred_pc = voxel2pc(pred.squeeze())
        # gt_pc = voxel2pc(gt.squeeze())
        # pred_pc = torch.Tensor(pred_pc).unsqueeze(0).cuda()
        # gt_pc = torch.Tensor(gt_pc).unsqueeze(0).cuda()
        
        pred_pc = torch.nonzero(pred.squeeze() >= 0.5).cuda()
        gt_pc = torch.nonzero(gt.squeeze() >= 0.5).cuda()
        # print(pred_pc.shape, gt_pc.shape)
        pred_pc = pred_pc.float().unsqueeze(0)
        gt_pc = gt_pc.float().unsqueeze(0)

        # normalization
        pred_pc /= 31
        gt_pc /= 31

        # calculate CD between preds and gt
        score = chamfer_dist(pred_pc, gt_pc)
        if score!=score:
            print(pred_pc.shape, gt_pc.shape)
        # d_score += score.detach().cpu().item()
    return score
    # return cd_score/batch_size


def evaluate_iou(preds, gt, thresh):
    # preds_occupy = (preds >= thresh).astype(np.float32)
    # diff = np.sum(np.logical_xor(preds_occupy, gt))
    # intersection = np.sum(np.logical_and(preds_occupy, gt))
    # union = np.sum(np.logical_or(preds_occupy, gt))
    # num_fp = np.sum(np.logical_and(preds_occupy, gt))  # false positive
    # num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt))  # false negative
    # return np.array([diff, intersection, union, num_fp, num_fn])
    # if np.sum(gt) == 0:
    #     print("?????")
    # if np.sum((preds >= thresh).astype(np.float32)) == 0:
    #     print("!!!!!!!!")
    preds_occupy = (preds >= thresh)
    gt_occupy = (gt >= thresh)
    intersection = np.sum(np.logical_and(preds_occupy, gt_occupy).astype(np.float32))
    union = np.sum(np.logical_or(preds_occupy, gt_occupy).astype(np.float32))
    return intersection, union


def voxel2pc(voxel):
    d, h, w = voxel.shape
    # point_num = 0
    points = []
    for k in range(d):
        for i in range(h):
            for j in range(w):
                if voxel[k][i][j] >= 0.5:
                    # point_num += 1
                    points.append([i, j, k])
    return points
# def voxel2pc(voxel):
#     b, d, h, w = voxel.shape
#     # point_num = 0
#     point_clouds = []
#     for b in range(b):
#         points = []
#         for k in range(d):
#             for i in range(h):
#                 for j in range(w):
#                     if voxel[b][k][i][j] >= 0.5:
#                         # point_num += 1
#                         points.append([i, j, k])
#         point_clouds.append(points)
#     return point_clouds


def voxel2mesh(voxels, surface_view):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > 0.3)
    voxels[positions] = 1
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face
        if not surface_view or np.sum(voxels[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]) < 27:
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)

    return np.array(verts), np.array(faces)


# pix3d scale (src) to shapenet scale (dst)
def rescale_pix3d_voxel(vxl_src,
                d_src=20.038700760193503, h_src=30.66620594333103, w_src=20.852107809260538,
                d_dst=15.18812246403541, h_dst=22.558465510881593, w_dst=15.704537071191442):
    assert d_src > d_dst
    assert h_src > h_dst
    assert w_src > w_dst
    d, h, w = vxl_src.shape

    d_pad = int(d * (d_src - d_dst) / float(d_dst))
    if d_pad % 2 == 1: d_pad = d_pad + 1
    h_pad = int(h * (h_src - h_dst) / float(h_dst))
    if h_pad % 2 == 1: h_pad = h_pad + 1
    w_pad = int(w * (w_src - w_dst) / float(w_dst))
    if w_pad % 2 == 1: w_pad = w_pad + 1

    vxl = np.zeros((d + d_pad, h + h_pad, w + w_pad), dtype=np.float32)
    vxl[int(d_pad / 2):int(d_pad / 2) + d, int(h_pad / 2):int(h_pad / 2) + h, int(w_pad / 2):int(w_pad / 2) + w] = vxl_src

    s_d = float(d_pad + d) / float(d)
    s_h = float(h_pad + h) / float(h)
    s_w = float(w_pad + w) / float(w)

    vxl_dst = np.zeros_like(vxl_src)
    for i in range(d):
        for j in range(h):
            for k in range(w):
                key_sum = np.sum(vxl[
                                 int(i * s_d):int((i + 1) * s_d),
                                 int(j * s_h):int((j + 1) * s_h),
                                 int(k * s_w):int((k + 1) * s_w)])
                vxl_dst[i, j, k] = 1 if key_sum >= 1.0 else 0

    return vxl_dst


def move_center(voxel):
    voxel = voxel.squeeze() #.detach().cpu()
    curr_points = torch.nonzero(voxel >= 0.5)
    if len(curr_points) == 0:
        return voxel

    curr_center = torch.sum(curr_points, axis=0).float() / len(curr_points)
    target_center = torch.Tensor([15.5, 15.5, 15.5]).cuda()
    mv_xyz = torch.round(target_center - curr_center).long()
    print(curr_center, mv_xyz)
    mx, my, mz = mv_xyz[0], mv_xyz[1], mv_xyz[2]
    if mx >= 32 or mx < -32:
        return voxel
    if my >= 32 or my < -32:
        return voxel
    if mz >= 32 or mz < -32:
        return voxel
    print('calibrate')
    voxel_x = torch.zeros_like(voxel).cuda()
    if mx >= 0:
        voxel_x[mx:, :, :] = voxel[:32-mx, :, :]
    else:
        voxel_x[:32+mx, :, :] = voxel[-mx:, :, :]
    
    voxel_xy = torch.zeros_like(voxel).cuda()
    if my >= 0:
        voxel_xy[:, my:, :] = voxel_x[:, :32-my, :]
    else:
        voxel_xy[:, :32+my, :] = voxel_x[:, -my:, :]
    
    voxel_xyz = torch.zeros_like(voxel).cuda()
    if mz >= 0:
        voxel_xyz[:, :, mz:] = voxel_xy[:, :, :32-mz]
    else:
        voxel_xyz[:, :, :32+mz] = voxel_xy[:, :, -mz:]

    return voxel_xyz
    

def voxel_downsampling(raw_vxl, factor=0.5):
    h, w, d = raw_vxl.shape

    dh, dw, dd = int(h * factor), int(w * factor), int(d * factor)
    dst_vxl = np.zeros((dh, dw, dd), dtype=np.int8)
    s = int(1 / factor)

    for i in range(dh):
        for j in range(dw):
            for k in range(dd):
                key_sum = np.sum(raw_vxl[i * s:(i + 1) * s, j * s:(j + 1) * s, k * s:(k + 1) * s])
                dst_vxl[i, j, k] = 1 if key_sum >= 1.0 else 0

    return dst_vxl


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))

# path: the path of dst .ply file
# voxel: the src voxel with shape [D, H, W]
def write_ply(path, voxel):
    points = voxel2pc(voxel)

    with open(path, 'w') as f:
        f.write("ply\r\n")
        f.write("format ascii 1.0\r\n")
        f.write("element vertex %d\r\n" % len(points))
        f.write("property float x\r\n")
        f.write("property float y\r\n")
        f.write("property float z\r\n")
        f.write("property uchar red\r\n")
        f.write("property uchar green\r\n")
        f.write("property uchar blue\r\n")
        f.write("end_header\r\n")

        for p in points:
            f.write("%f %f %f " %(p[0], p[1], p[2]))
            f.write("255 0 0\r\n")

        f.write("\r\n")


def voxel2obj(filename, pred, surface_view=True):
    verts, faces = voxel2mesh(pred, surface_view)
    write_obj(filename, verts, faces)



def create_opt(opt, init_lr, warm_epoch, lr_step=1, lr_gamma=0.9):
    
    warmup = GradualWarmupScheduler(opt, multiplier=1, total_epoch=warm_epoch,  
                    after_scheduler=StepLR(opt, step_size=lr_step, gamma=lr_gamma))
    opt.zero_grad()
    opt.step()
    return warmup


# def create_wp(opt, init_lr, warm_epoch, lr_step=1, lr_gamma=0.8):
    
#     warmup = GradualWarmupScheduler(opt, multiplier=1, total_epoch=warm_epoch,  
#                     after_scheduler=StepLR(opt, step_size=lr_step, gamma=lr_gamma))
#     opt.zero_grad()
#     opt.step()
#     return warmup


def set_grad(module, grad=True):
    for param in module.parameters():
        param.requires_grad = grad
