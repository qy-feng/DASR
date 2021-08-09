import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/data/qianyu/3d/PATR')
from models.backbone.resnet import resnet18, resnet34, resnet50
# from models.attention.afnb import AFNB
from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Encoder(nn.Module):
    def __init__(self, out_dim, resnet_vs='50'):
        super(Encoder, self).__init__()

        if resnet_vs == '50':
            self.backbone = resnet50(num_classes=out_dim)
        else:
            self.backbone = resnet34(num_classes=out_dim)
        self.load_2d_pretrain(version=resnet_vs, freeze_lnum=7)

    def load_2d_pretrain(self, version='50', freeze_lnum=0, progress=True):
        model_dict = self.backbone.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls['resnet%s'%version], progress=progress)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.backbone.load_state_dict(model_dict)
        
        if freeze_lnum:
            ct = 0
            for child in self.backbone.children():
                if ct < freeze_lnum:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1
    
    def forward(self, x, wf=False):
        return self.backbone(x, wf=wf)


class EncoderAtt(nn.Module):
    def __init__(self, out_dim, resnet_vs='50'):
        super(EncoderAtt, self).__init__()

        if resnet_vs == '50':
            self.backbone = resnet50(num_classes=out_dim)
        else:
            self.backbone = resnet34(num_classes=out_dim)
        self.load_2d_pretrain(version=resnet_vs, freeze_lnum=7)

        self.att_fusion = AFNB(1024, 2048, 2048, 256, 256, 
                               dropout=0.05, sizes=([1]), 
                               norm_type="encsync_batchnorm")
        self.dense = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, 
                               padding=0, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.my_fc = nn.Linear(1024, out_dim)
        
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)

    def load_2d_pretrain(self, version='50', freeze_lnum=0, progress=True):
        model_dict = self.backbone.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls['resnet%s'%version], progress=progress)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.backbone.load_state_dict(model_dict)
        
        if freeze_lnum:
            ct = 0
            for child in self.backbone.children():
                if ct < freeze_lnum:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1
    
    def forward(self, x, wf=False):
        
        # non-local
        x_, feats = self.backbone(x, wf=True)
        x = self.att_fusion(feats[-2], feats[-1])

        x = self.dense(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.my_fc(x)

        x = x + x_
        x = self.layer_norm(x)
        if wf:
            return x, feats
        else:
            return x

