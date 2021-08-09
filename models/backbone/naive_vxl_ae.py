# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_Naive_voxel(nn.Module):
    def __init__(self, ef_dim, code_len, use_att=False):
        super(Encoder_Naive_voxel, self).__init__()

        self.ef_dim = ef_dim
        self.use_att = use_att

        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=1, bias=False)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=False)  # 16
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 3, stride=1, padding=1, bias=False)  # 8
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 3, stride=1, padding=1, bias=False)  # 4
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 3, stride=1, padding=1, bias=False)  # 2
        self.code = nn.Linear(2 * 2 * 2 * (self.ef_dim * 8), code_len, bias=True)
        nn.init.xavier_uniform_(self.conv_0.weight)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.code.weight)

    def forward(self, inputs):

        d_0 = self.conv_0(inputs)
        d_0 = F.relu(d_0)

        d_1 = self.conv_1(d_0)
        d_1 = F.max_pool3d(d_1, 2, 2)
        d_1 = F.relu(d_1)

        d_2 = self.conv_2(d_1)
        d_2 = F.max_pool3d(d_2, 2, 2)
        d_2 = F.relu(d_2)

        d_3 = self.conv_3(d_2)
        d_3 = F.max_pool3d(d_3, 2, 2)
        d_3 = F.relu(d_3)

        d_4 = self.conv_4(d_3)
        d_4 = F.max_pool3d(d_4, 2, 2)
        d_4 = F.relu(d_4)
        code = self.code(d_4.view(d_4.shape[0], -1))

        # add sigmoid
        # code = F.relu(code)
        # code = torch.sigmoid(code)

        if not self.use_att:
            return code
        else:
            return code, d_4


class Decoder_Naive_voxel(nn.Module):
    def __init__(self, ef_dim, code_len):
        super(Decoder_Naive_voxel, self).__init__()
        self.ef_dim = ef_dim
        self.conv_0 = nn.Conv3d(self.ef_dim * 8, self.ef_dim * 8, 3, stride=1, padding=1, bias=False) # 2
        self.conv_1 = nn.Conv3d(self.ef_dim * 8, self.ef_dim * 4, 3, stride=1, padding=1, bias=False)  # 4
        self.conv_2 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 2, 3, stride=1, padding=1, bias=False)  # 8
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim, 3, stride=1, padding=1, bias=False)  # 16
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=False)  # 32
        self.out_conv = nn.Conv3d(self.ef_dim, 1, 3, stride=1, padding=1, bias=False)
        self.decode = nn.Linear(code_len, 2 * 2 * 2 * (self.ef_dim * 8), bias=True)
        nn.init.xavier_uniform_(self.conv_0.weight)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.out_conv.weight)
        nn.init.xavier_uniform_(self.decode.weight)

    def forward(self, laten_vxl):

        code = self.decode(laten_vxl)
        code = F.relu(code)

        d_0 = self.conv_0(code.view(code.shape[0], self.ef_dim * 8, 2, 2, 2)) # 2
        d_0 = F.relu(d_0)
        d_0 = F.interpolate(d_0, scale_factor=2.0) # 4

        d_1 = self.conv_1(d_0)
        d_1 = F.relu(d_1)
        d_1 = F.interpolate(d_1, scale_factor=2.0) # 8

        d_2 = self.conv_2(d_1)
        d_2 = F.relu(d_2)
        d_2 = F.interpolate(d_2, scale_factor=2.0) # 16

        d_3 = self.conv_3(d_2)
        d_3 = F.relu(d_3)
        d_3 = F.interpolate(d_3, scale_factor=2.0) # 32

        d_4 = self.conv_4(d_3)
        d_4 = F.relu(d_4)

        out = self.out_conv(d_4)
        out = torch.sigmoid(out)

        return out


class VoxelNaiveAE(nn.Module):
    def __init__(self, ef_dim, code_len):
        super(VoxelNaiveAE, self).__init__()
        self.encoder = Encoder_Naive_voxel(ef_dim, code_len)
        self.decoder = Decoder_Naive_voxel(ef_dim, code_len)

    def forward(self, voxels):
        vxl = self.decoder(self.encoder(voxels))
        return vxl
