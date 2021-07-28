import os
import numpy as np
import time
import argparse
import csv
from tqdm import tqdm
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import StepLR, ExponentialLR
# from warmup_scheduler import GradualWarmupScheduler
# import pytorch_warmup as warmup
from utils import *
from test import test

warm_epoch = 1
log_interval = 200
unf_enc = -1
unf_dec = 10
with_SCM = False


def train(mdl, train_loader, test_loader, device, args):
    print("Start training...")
    init_lr = args.lr
    num_steps = len(train_loader) * args.epoch_num
    opt_f = torch.optim.Adam(mdl.f.parameters(), lr=init_lr)
    warmup_f = create_opt(opt_f, args.lr, warm_epoch)

    opt_dd = torch.optim.Adam(mdl.d_domain.parameters(), lr=init_lr)
    warmup_dd = create_opt(opt_dd, args.lr, warm_epoch)

    opt_ds = torch.optim.Adam(mdl.d_shape.parameters(), lr=init_lr)
    warmup_ds = create_opt(opt_ds, args.lr, warm_epoch)

    lambda_d, lambda_s = args.lamb_d, args.lamb_s
    best_iou, best_iter = 0, 0

    start_time = time.time()
    iter_num = 0
    for epoch in range(args.epoch_num):

        # adjust learning rate
        warmup_f.step()
        warmup_dd.step()
        warmup_ds.step()
        lr = args.lr * 0.9 **(epoch-warm_epoch) if epoch >=warm_epoch else args.lr

        mdl.train()
        for i, batch_data in tqdm(enumerate(train_loader)):
            s_img = batch_data['s_img'].cuda()
            s_voxel = batch_data['s_voxel'].cuda()
            r_img = batch_data['r_img'].cuda()
            pos_label = torch.full((args.batch_size, 1), 1, dtype=torch.float32).cuda()
            neg_label = torch.full((args.batch_size, 1), 0, dtype=torch.float32).cuda()
            sim_label = batch_data['sim_label'].cuda()
            
            # unfreeze modules
            if iter_num // log_interval == unf_enc:
                print('unfreeze resnet')
                set_grad(mdl.f, grad=True)

            if iter_num // log_interval == unf_dec:
                print('unfreeze 3d decoder')
                set_grad(mdl.decoder, grad=True)

            ############## d_shape (2D|3D) ##############
            opt_ds.zero_grad()
            errD_s = mdl.adapt_shape(s_img, s_voxel,
                                     [pos_label, neg_label])
            ds_loss = errD_s.item()

            errD_s = errD_s * lambda_s
            errD_s.backward()
            opt_ds.step()

            ############## d_domain (Syn|Real) ##############
            opt_dd.zero_grad()
            errD_d = mdl.adapt_domain(s_img, r_img,
                                     [pos_label, neg_label])
            dd_loss = errD_d.item()

            errD_d = errD_d * lambda_d
            errD_d.backward()
            opt_dd.step()

            ################## f ##################

            opt_f.zero_grad()
            errR = mdl.reconstruct(s_img, s_voxel)

            rec_loss = errR.item()

            if with_SCM:
                err_C = mdl.contrast(s_img, r_img, s_voxel, sim_label)
                errR += err_C * 1e-3

            errG_s = mdl.adapt_shape(s_img, s_voxel,
                                     [neg_label, pos_label])
            errG_d = mdl.adapt_domain(s_img, r_img,
                                     [neg_label, pos_label])
            errR += errG_s * lambda_s
            errR += errG_d * lambda_d

            errR.backward()
            opt_f.step()

            iter_num += 1

            if iter_num % log_interval == 0:

                print("\nTrain: Iter %d (epoch %d LR: %.7f time: %.4f) loss_dd: %.6f, loss_ds: %.6f, loss_rec: %.6f" % (
                    iter_num, epoch, lr, time.time() - start_time,
                    dd_loss, ds_loss, rec_loss))

                valid_loss, valid_mean_iou = test(mdl, test_loader, device)
                print("\n%s (valid) Epoch: %d Iter: %d loss_rec: %.6f, iou: %.6f\n" % (
                    args.exp_name, epoch, iter_num, valid_loss, valid_mean_iou))

                # save checkpoint
                if valid_mean_iou > best_iou:
                    best_iter = iter_num
                    best_iou = valid_mean_iou
                    save_dir = os.path.join(args.checkpoint_dir, "best_model.pth")
                    torch.save(mdl.state_dict(), save_dir)

                with open(os.path.join(args.checkpoint_dir, 'train_valid.csv'), 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([iter_num, rec_loss/(i+1), valid_loss, valid_mean_iou, best_iter, round(best_iou, 5)])
        
                if iter_num // log_interval > unf_dec:
                    save_dir = os.path.join(args.checkpoint_dir, "iter_%d.pth" % iter_num)
                    torch.save(mdl.state_dict(), save_dir)

    print('Training finished. Best iter: %d, best IoU: %.4f' % (best_iter, best_iou))
