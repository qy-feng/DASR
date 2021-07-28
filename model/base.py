# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from .backbone.ae_3d import *
from .backbone.naive_vxl_ae import *
from .enc import Encoder
from .disc import Discriminator_domain, Discriminator_shape
from .backbone.resnet import my_resnet, resnet18, resnet34, resnet50
from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

class SVDA(nn.Module):
    def __init__(self, args,
                 ef_dim=32, 
                 z_dim=256):
        super(SVDA, self).__init__()
        
        self.f = resnet50(num_classes=z_dim)
        self.load_2d_pretrain(version='50', freeze_lnum=7)
        
        if args.implicit:
            self.encoder = Encoder_voxel(ef_dim, z_dim)
            self.decoder = Decoder_voxel(ef_dim, z_dim)
            emb_dim = z_dim
        else:
            self.encoder = Encoder_Naive_voxel(ef_dim, z_dim)
            self.decoder = Decoder_Naive_voxel(ef_dim, z_dim)
            emb_dim = ef_dim
        self.load_3d_pretrain(args.pretrain_path)
        
        self.d_domain = Discriminator_domain(z_dim)
        self.d_shape = Discriminator_shape(z_dim)

        self.crit_bce = nn.BCELoss()

    def load_2d_pretrain(self, version='50', freeze_lnum=0, progress=True):
        model_dict = self.f.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls['resnet%s'%version], progress=progress)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.f.load_state_dict(model_dict)
        
        if freeze_lnum:
            ct = 0
            for child in self.f.children():
                if ct < freeze_lnum:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1

    def load_3d_pretrain(self, pretrain_path):
        pretrained_dict = torch.load(pretrain_path)

        E_dict = self.encoder.state_dict()
        pretrained_E_dict = dict()
        for k, v in E_dict.items():
            if 'encoder.'+k in pretrained_dict:
                pretrained_E_dict[k] = pretrained_dict['encoder.'+k]
        print('load', pretrained_E_dict.keys())
        E_dict.update(pretrained_E_dict) 
        self.encoder.load_state_dict(E_dict)

        D_dict = self.decoder.state_dict()
        pretrained_D_dict = dict()
        for k, v in D_dict.items():
            if 'decoder.'+k in pretrained_dict:
                pretrained_D_dict[k] = pretrained_dict['decoder.'+k]
        D_dict.update(pretrained_D_dict) 
        self.decoder.load_state_dict(D_dict)
        # freeze E D
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def adapt_shape(self, s_img, s_voxel, labels):
        s_img_enc = self.f(s_img)
        s_img_shape = self.d_shape(s_img_enc)
        s_vxl_enc = self.encoder(s_voxel)
        s_vxl_shape = self.d_shape(s_vxl_enc)
        
        errD_img = self.crit_bce(s_img_shape, labels[0])
        errD_vxl = self.crit_bce(s_vxl_shape, labels[1])
        errD_s = errD_img + errD_vxl
        return errD_s

    def adapt_domain(self, s_img, r_img, labels):
        s_img_enc = self.f(s_img)
        s_img_domain = self.d_domain(s_img_enc)
        r_img_enc = self.f(r_img)
        r_img_domain = self.d_domain(r_img_enc)
        
        errD_img = self.crit_bce(s_img_domain, labels[0])
        errD_vxl = self.crit_bce(r_img_domain, labels[1])
        errD_d = errD_img + errD_vxl
        return errD_d

    def reconstruct(self, s_img, s_voxel):
        s_img_enc = self.f(s_img)
        s_img_rec = self.decoder(s_img_enc)

        errR = self.crit_bce(s_img_rec, s_voxel)
        return errR

    def forward(self, s_img, s_vxl, r_img):
        # synthetic img
        embed_s_img = self.f(s_img)
        # embed_s_img, s_att = self.f(s_img)
        rec_s_img = self.decoder(embed_s_img)

        # domain_s_img = self.d_domain(embed_s_img)
        # shape_s_img = self.d_shape(embed_s_img)

        # synthetic voxel
        embed_s_vxl = self.encoder(s_vxl)
        # print(embed_s_vxl.shape)
        # shape_s_vxl = self.d_shape(embed_s_vxl)

        # real img
        embed_r_img = self.f(r_img)
        # embed_r_img, r_att = self.f(r_img)
        rec_r_img = self.decoder(embed_r_img)
        # domain_r_img = self.d_domain(embed_r_img)
        # rec_r_img = self.decoder(embed_r_img)
        return rec_s_img, rec_r_img, embed_s_vxl

