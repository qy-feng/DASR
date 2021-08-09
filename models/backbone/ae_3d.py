# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_voxel(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(Encoder_voxel, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)  # 16
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=False)  # 8
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=False)  # 4
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=False)  # 2
        self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.z_dim, 4, stride=2, padding=1, bias=True)  # 1
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.z_dim)
        d_5 = torch.sigmoid(d_5)

        return d_5


class Decoder_voxel(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(Decoder_voxel, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.conv_1 = nn.ConvTranspose3d(self.z_dim, self.ef_dim * 8, 4, stride=2, padding=1, bias=False)  # 2
        self.in_1 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_2 = nn.ConvTranspose3d(self.ef_dim * 8, self.ef_dim * 4, 4, stride=2, padding=1, bias=False)  # 4
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_3 = nn.ConvTranspose3d(self.ef_dim * 4, self.ef_dim * 2, 4, stride=2, padding=1, bias=False)  # 8
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_4 = nn.ConvTranspose3d(self.ef_dim * 2, self.ef_dim, 4, stride=2, padding=1, bias=False)  # 16
        self.in_4 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_5 = nn.ConvTranspose3d(self.ef_dim, 1, 4, stride=2, padding=1, bias=True)  # 32
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, laten_vector):
        d_1 = self.in_1(self.conv_1(laten_vector.view(-1, self.z_dim, 1, 1, 1)))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = torch.sigmoid(d_5)

        return d_5


class VoxelAE(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(VoxelAE, self).__init__()
        self.encoder = Encoder_voxel(ef_dim, z_dim)
        self.decoder = Decoder_voxel(ef_dim, z_dim)

    def forward(self, voxels):
        laten_vecs = self.encoder(voxels)
        # return laten_vecs
        return self.decoder(laten_vecs)


# former implementation of f
class Encoder_2d(nn.Module):
    def __init__(self, out_dim=128, ndf=4):
        super(Encoder_2d, self).__init__()
        self.enc_layers = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(1, ndf//2 , 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf/2) x 128 x 128
            nn.Conv2d(ndf//2, ndf, 4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            # nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
        )
        self.final_layer = nn.Linear(1568, out_dim)

    def forward(self, imgs):
        feats = self.enc_layers(imgs) 
        out = torch.flatten(feats, start_dim=1)
        out = self.final_layer(out)
        return out


if __name__ == '__main__':

    # test 3d
    samples = torch.randn((8, 1, 32, 32, 32))
    #vxl_ae = VoxelAE(32, 128)
    #pd = vxl_ae(samples)
    #print(pd.shape)
    
    # test 2d
    #samples = torch.randn((8, 3, 256, 256))
    #enc_2d = Encoder_2d()
    #pd = enc_2d(samples)
    #print(pd.shape)
