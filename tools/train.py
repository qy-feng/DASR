import os
import numpy as np
import time
import csv
from tqdm import tqdm
import torch
from utils import *
from config import parse
import sys
sys.path.append('/data/qianyu/3d/DA-vx')
from dataloader.my_loader_pix import load_data as load_pix
from dataloader.my_loader_psc import load_data as load_psc
from models.model import LMM
from test import test
warm_epoch = 1
log_interval = 200
unf_enc = -1
unf_dec = 10
with_SCM = False


def train(model, train_loader, test_loader, device, args):

    init_lr = args.lr
    # num_steps = len(train_loader) * args.epoch_num
    opt_f = torch.optim.Adam(model.f.parameters(), lr=init_lr)
    # warmup_f = create_opt(opt_f, args.lr, warm_epoch)
    opt_d = torch.optim.Adam(model.d_domain.parameters(), lr=init_lr)
    # warmup_dd = create_opt(opt_d, args.lr, warm_epoch)
    opt_s = torch.optim.Adam(model.d_shape.parameters(), lr=init_lr)
    # warmup_ds = create_opt(opt_s, args.lr, warm_epoch)

    lambda_d, lambda_s = args.lamb_d, args.lamb_s
    best_iou, best_iter = 0, 0

    start_time = time.time()
    iter_num = 0
    for epoch in range(args.epoch_num):
        # adjust learning rate
        opt_f.step()
        opt_d.step()
        opt_s.step()
        lr = args.lr * 0.9 **(epoch-warm_epoch) if epoch >=warm_epoch else args.lr

        model.train()
        for i, batch_data in tqdm(enumerate(train_loader)):
            s_img_pos = batch_data['s_img_pos'].cuda()
            s_img_neg = batch_data['s_img_neg'].cuda()
            s_vxl_pos = batch_data['s_vxl_pos'].cuda()
            s_vxl_neg = batch_data['s_vxl_neg'].cuda()
            r_img = batch_data['r_img'].cuda()

            ############## shape (2D|3D) ##############
            opt_s.zero_grad()
            errS = model.adapt_shape(s_img_pos, s_vxl_pos)
            errS += model.adapt_shape(s_img_neg, s_vxl_neg) 

            errS *= lambda_s
            errS.backward()
            opt_s.step()

            ############## domain (Syn|Real) ##############
            opt_d.zero_grad()
            errD = model.adapt_domain(s_img_pos, r_img)

            errD = errD * lambda_d
            errD.backward()
            opt_d.step()

            ################## f ##################
            opt_f.zero_grad()
            rec_pos, err_pos = model.reconstruct(s_img_pos, s_vxl_pos)
            rec_neg, err_neg = model.reconstruct(s_img_neg, s_vxl_neg)
            errR = err_pos + err_neg
            # if with_SCM:
            #     err_C = model.contrast(s_img, r_img, s_voxel, sim_label)
            #     errR += err_C * 1e-3

            errS_adv = model.adapt_shape(s_vxl_pos, s_img_pos)
            errS_adv += model.adapt_shape(s_vxl_neg, s_img_neg)
            errD_adv = model.adapt_domain(r_img, s_img_pos)
            errR += errS_adv * lambda_s
            errR += errD_adv * lambda_d

            # margin loss
            anchor = model.f(r_img)
            pos_f = model.f(s_img_pos)
            neg_f = model.f(s_img_neg)
            # fst_loss = first_order(anchor, pos_f, neg_f)

            pos_v = model.encode(s_vxl_pos)
            neg_v = model.encode(s_vxl_neg)
            # scd_loss = second_order(pos_f, anchor, pos_v, rec_pos)
            # scd_loss += second_order(neg_f, anchor, neg_v, rec_neg)

            errR.backward()
            opt_f.step()

            iter_num += 1
            shape_loss = errD.item()
            dom_loss = errD.item()
            rec_loss = errR.item()
            if iter_num % log_interval == 0:

                print("\nTrain: Iter %d (epoch %d LR: %.7f time: %.4f) loss_dd: %.6f, loss_ds: %.6f, loss_rec: %.6f" % (
                    iter_num, epoch, lr, time.time() - start_time,
                    shape_loss, domain_loss, rec_loss))

                valid_loss, valid_mean_iou = test(model, test_loader, device)
                print("\n%s (valid) Epoch: %d Iter: %d loss_rec: %.6f, iou: %.6f\n" % (
                    args.exp_name, epoch, iter_num, valid_loss, valid_mean_iou))

                # save checkpoint
                if valid_mean_iou > best_iou:
                    best_iter = iter_num
                    best_iou = valid_mean_iou
                    save_dir = os.path.join(args.checkpoint_dir, "best_model.pth")
                    torch.save(model.state_dict(), save_dir)

                with open(os.path.join(args.checkpoint_dir, 'train_valid.csv'), 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([iter_num, rec_loss/(i+1), valid_loss, valid_mean_iou, best_iter, round(best_iou, 5)])
        
                if iter_num // log_interval > unf_dec:
                    save_dir = os.path.join(args.checkpoint_dir, "iter_%d.pth" % iter_num)
                    torch.save(model.state_dict(), save_dir)

    print('Training finished. Best iter: %d, best IoU: %.4f' % (best_iter, best_iou))


if __name__ == '__main__':
    args = parse()

    # environment setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    project_dir = "/data/qianyu/3d/DA-vx"

    # implicit model
    if args.implicit:
        print('Use implicit 3D backbone')
        args.pretrain_path = 'vxl_ae_best.pth'
    args.pretrain_path = os.path.join("checkpoints", args.pretrain_path)

    # create checkpoint dir
    args.checkpoint_dir = os.path.join("checkpoints", args.exp_name)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # logging file
    with open(os.path.join(args.checkpoint_dir, 'train_valid.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Iter', 'train_loss', 'valid_loss', 'valid_iou', 'best_iter', 'best_iou'])

    # load data
    if args.test_data == 'pixel':
        train_loader, test_loader = load_pix(args)
    else:
        train_loader, test_loader = load_psc(args)

    # construct model
    if args.baseline:
        mdl = SVDA(args).cuda()
    else:
        mdl = LMM(args).cuda()

    if args.resume_path:
        print('Resume from', args.resume_path)
        mdl.load_state_dict(torch.load(os.path.join('checkpoints', args.resume_path)))

    if args.eval:
        loss, iou = test(mdl, test_loader, device)
        print('Exp %s Test result: loss %4f iou %4f' % (args.exp_name, loss, iou))
    else:
        train(mdl, train_loader, test_loader, device, args)
