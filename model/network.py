# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
from .backbone.ae_3d import *
from .backbone.naive_vxl_ae import *
from .enc import Encoder, EncoderAtt
from .disc import Discriminator_domain, Discriminator_shape
from .modules.afnb import SelfAttentionBlock2D
with_SSA = True

class LMM(nn.Module):
    def __init__(self, args,
                 ef_dim=32, 
                 z_dim=256):
        super(LMM, self).__init__()

        if with_SSA:
            self.f = EncoderAtt(z_dim)
        else:
            self.f = Encoder(z_dim)
        
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
        heads = 8
        self.d_att = CrossAttention(z_dim, heads)
        self.final_linear = nn.Linear(heads, 1)
        self.crit_bce = nn.BCELoss()

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
        # s_img_enc, feats = self.f(s_img, wf=True)
        s_img_rec = self.decoder(s_img_enc)
        errR = self.crit_bce(s_img_rec, s_voxel)

        return errR

    def contrast(self, s_img, r_img, s_voxel, sim_label):
        s_img_enc = self.f(s_img)
        # s_img_enc, feats = self.f(s_img, wf=True)
        r_img_enc = self.f(r_img)
        s_vxl_enc = self.encoder(s_voxel)

        # cossim = nn.CosineSimilarity()
        # sim_label = cossim(s_img_enc, r_img_enc).cpu().detach().numpy()
        # sim_label = torch.Tensor(sim_label).cuda()
        # print(sim_label.shape)
        # pos_num = torch.sum(sim_label >= 0.6)
        # print('positive', pos_num)

        _, sim_map = self.d_att(r_img_enc, s_vxl_enc, s_vxl_enc)
        sim_map = sim_map.squeeze()
        sim_map = self.final_linear(sim_map)
        # print(sim_map.shape, sim_label.shape)
        err = self.crit_bce(sim_map, sim_label)

        return err


class ScaleDotProductAttention(nn.Module):
  
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        _, _, _, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2,3) # view(batch_size, head, d_tensor, length) 
        # print(q.shape, k_t.shape)
        score = torch.matmul(q, k_t) / math.sqrt(d_tensor) 

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        att = F.softmax(score, dim=-1)

        # 4. multiply with Value
        v = torch.matmul(att, v)

        return v, att


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_head=4, drop_prob=0.1):
        super(CrossAttention, self).__init__()

        self.model_dim = d_model
        self.n_head = n_head
        self.head_dim = self.model_dim // self.n_head

        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head) 
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head) 
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head)

        self.linear_final=nn.Linear(self.head_dim * self.n_head, self.model_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.scaled_dot_product_attention = ScaleDotProductAttention()


    def forward(self, query, key, value, mask=None):
        batch_size = key.size(0)
        len_q, len_k, len_v = query.size(1), key.size(1), value.size(1)

        q = self.linear_q(query) 
        k = self.linear_k(key)
        v = self.linear_v(value)
        
        # q_ = q.view(batch_size * self.n_head, -1, self.head_dim) 
        # k_ = k.view(batch_size * self.n_head, -1, self.head_dim)
        # v_ = v.view(batch_size * self.n_head, -1, self.head_dim)
        q_ = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k_ = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v_ = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)

        context, att = self.scaled_dot_product_attention(q_, k_, v_, mask) 
        context = context.transpose(1, 2)
        
        return context, att
        # output = context.reshape(batch_size, -1, self.head_dim * self.n_head) 
        # output = self.linear_final(output)
        # output = self.dropout(output)
        # return output


################## with attention ##################
# modes = ['spatial', 'channel', 'joint']
# # add_transformer = True
# add_transformer = False

# class SADA(nn.Module):
#     def __init__(self, pretrain_path=None, naive=True,
#                  ef_dim=32, z_dim=256, att_mode=0):
#         super(SADA, self).__init__()
        
#         if naive:
#             self.encoder = Encoder_Naive_voxel(ef_dim, z_dim, use_att=True)
#             self.decoder = Decoder_Naive_voxel(ef_dim, z_dim)
#             emb_dim = ef_dim
#         else:
#             self.encoder = Encoder_voxel(ef_dim, z_dim)
#             self.decoder = Decoder_voxel(ef_dim, z_dim)
#             emb_dim = z_dim

#         resnet_vs = '34'
#         self.f = bam_resnet34(attention=modes[att_mode], num_classes=z_dim)
#         self.load_2d_pretrain(version=resnet_vs, freeze_lnum=6)

#         self.d_domain = Discriminator_domain(z_dim)
#         self.d_shape = Discriminator_shape(z_dim)
 
#         if pretrain_path:
#             self.load_3d_pretrain(pretrain_path)

#         # freeze E D
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         for param in self.decoder.parameters():
#             param.requires_grad = False

#         self.crit = nn.BCELoss()

#         # transformer
#         if add_transformer:
#             self.trans_shp = Transformer()
#             self.trans_dmn = Transformer()

#     def load_2d_pretrain(self, version='50', freeze_lnum=0, progress=True):
        
#         model_dict = self.f.state_dict()
#         pretrained_dict = load_state_dict_from_url(model_urls['resnet%s'%version], progress=progress)
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         model_dict.update(pretrained_dict) 
#         self.f.load_state_dict(model_dict)
        
#         if freeze_lnum:
#             ct = 0
#             for child in self.f.children():
#                 if ct < freeze_lnum:
#                     for param in child.parameters():
#                         param.requires_grad = False
#                 ct += 1

#     def load_3d_pretrain(self, pretrain_path):
#         pretrained_dict = torch.load(pretrain_path)

#         E_dict = self.encoder.state_dict()
#         pretrained_E_dict = dict()
#         for k, v in E_dict.items():
#             if 'encoder.'+k in pretrained_dict:
#                 pretrained_E_dict[k] = pretrained_dict['encoder.'+k]
#         print('load', pretrained_E_dict.keys())
#         E_dict.update(pretrained_E_dict) 
#         self.encoder.load_state_dict(E_dict)

#         D_dict = self.decoder.state_dict()
#         pretrained_D_dict = dict()
#         for k, v in D_dict.items():
#             if 'decoder.'+k in pretrained_dict:
#                 pretrained_D_dict[k] = pretrained_dict['decoder.'+k]
#         D_dict.update(pretrained_D_dict) 
#         self.decoder.load_state_dict(D_dict)

#     def adapt_shape(self, s_img, s_voxel, labels):
#         s_vxl_enc, s_vxl_tgt = self.encoder(s_voxel)
#         s_img_enc, s_img_att = self.f(s_img, s_vxl_tgt)

#         s_img_shape = self.d_shape(s_img_enc)
#         s_vxl_shape = self.d_shape(s_vxl_enc)
        
#         if add_transformer:
#             s_img_att = self.trans_shp(s_img_att, s_vxl_tgt)
#             # TODO: use s_img_att to calculate s_img_enc

#         errD_img = self.crit(s_img_shape, labels[0])
#         errD_vxl = self.crit(s_vxl_shape, labels[1])
#         errD_s = errD_img + errD_vxl
#         return errD_s, s_img_att, s_vxl_enc

#     def adapt_domain(self, s_img, r_img, labels):
#         s_img_enc, s_img_att = self.f(s_img)
#         s_img_domain = self.d_domain(s_img_enc)
#         r_img_enc, r_img_att = self.f(r_img)
#         r_img_domain = self.d_domain(r_img_enc)
        
#         # TODO: add channel-wise attention to r_img
#         # if add_transformer:
#         #     r_img_att = self.trans_dmn(r_img_att, s_img_att)

#         errD_img = self.crit(s_img_domain, labels[0])
#         errD_vxl = self.crit(r_img_domain, labels[1])
#         errD_d = errD_img + errD_vxl
#         return errD_d, s_img_att, r_img_att

#     def reconstruct(self, s_img, s_voxel):
#         s_img_enc, _ = self.f(s_img)
#         s_img_rec = self.decoder(s_img_enc)
#         errR = self.crit(s_img_rec, s_voxel)
#         return errR
