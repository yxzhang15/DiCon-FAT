#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks import ProjHead
from model_utils.FreDe import FreDe, select_top_20_percent_from_all34
      
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class get_model(nn.Module):
    def __init__(self, output_channels=40, k=20, emb_dims=1024, dropout=0.5, normal_channel=True, use_pre_defense=False):
        super(get_model, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.use_pre_defense = use_pre_defense

    def set_pre_head(self, pre_defense_head):
        # if we need preprocess-based defense module
        assert pre_defense_head is not None
        self.pre_defense_head = pre_defense_head

    def trans_multi(self, ori, indices, indicesl, indicesm, M, d):
        ori = ori.transpose(1,2).contiguous()
        batch_size = ori.size(0)
        num_points = ori.size(1)
        orih = ori.view(batch_size * num_points, -1)[indices, :]
        oril = ori.view(batch_size * num_points, -1)[indicesl, :]
        orim = ori.view(batch_size * num_points, -1)[indicesm, :]
        
        orih = orih.view(batch_size, M, -1)  # b,M,c
        oril = oril.view(batch_size, M, -1)  # b,M,c
        orim = orim.view(batch_size, M, -1)  # b,M,c
        
        orih_hard = select_top_20_percent_from_all34(orih, ori, batch_size)
        oril_hard = select_top_20_percent_from_all34(oril, ori, batch_size)
        orim_hard = select_top_20_percent_from_all34(orim, ori, batch_size)
        
        x1h = F.adaptive_max_pool1d(orih.permute(0,2,1), 1).view(batch_size, -1)
        x2h = F.adaptive_avg_pool1d(orih.permute(0,2,1), 1).view(batch_size, -1)
        orih = torch.cat((x1h, x2h), 1)
        
        x1m = F.adaptive_max_pool1d(orim.permute(0,2,1), 1).view(batch_size, -1)
        x2m = F.adaptive_avg_pool1d(orim.permute(0,2,1), 1).view(batch_size, -1)
        orim = torch.cat((x1m, x2m), 1)

        x1l = F.adaptive_max_pool1d(oril.permute(0,2,1), 1).view(batch_size, -1)
        x2l = F.adaptive_avg_pool1d(oril.permute(0,2,1), 1).view(batch_size, -1)
        oril = torch.cat((x1l, x2l), 1)

        x1h = F.adaptive_max_pool1d(orih_hard.permute(0,2,1), 1).view(batch_size, -1)
        x2h = F.adaptive_avg_pool1d(orih_hard.permute(0,2,1), 1).view(batch_size, -1)
        orih_hard = torch.cat((x1h, x2h), 1)
        
        x1m = F.adaptive_max_pool1d(orim_hard.permute(0,2,1), 1).view(batch_size, -1)
        x2m = F.adaptive_avg_pool1d(orim_hard.permute(0,2,1), 1).view(batch_size, -1)
        orim_hard = torch.cat((x1m, x2m), 1)

        x1l = F.adaptive_max_pool1d(oril_hard.permute(0,2,1), 1).view(batch_size, -1)
        x2l = F.adaptive_avg_pool1d(oril_hard.permute(0,2,1), 1).view(batch_size, -1)
        oril_hard = torch.cat((x1l, x2l), 1)
        
        return orih, oril, orim, orih_hard, orim_hard, oril_hard

    def forward(self, x, mask_ratio=None):
        num_points=x.size(2)
        xh, xm, xl = FreDe(x, num_points // 3)
        if self.use_pre_defense:
            assert self.pre_defense_head is not None
            x = self.pre_defense_head(x)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x_net4h, x_net4l, x_net4m, orih_hard, orim_hard, oril_hard = self.trans_multi(x, xh, xl, xm, num_points // 3, 1024)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)#20 2024
        gf=x

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        
        x_net4h = F.leaky_relu(self.bn6(self.linear1(x_net4h)), negative_slope=0.2)
        x_net4h = self.dp1(x_net4h)
        x_net4h = F.leaky_relu(self.bn7(self.linear2(x_net4h)), negative_slope=0.2)
        x_net4h = self.dp2(x_net4h)
        
        x_net4m = F.leaky_relu(self.bn6(self.linear1(x_net4m)), negative_slope=0.2)
        x_net4m = self.dp1(x_net4m)
        x_net4m = F.leaky_relu(self.bn7(self.linear2(x_net4m)), negative_slope=0.2)
        x_net4m = self.dp2(x_net4m)
    
        x_net4l = F.leaky_relu(self.bn6(self.linear1(x_net4l)), negative_slope=0.2)
        x_net4l = self.dp1(x_net4l)
        x_net4l = F.leaky_relu(self.bn7(self.linear2(x_net4l)), negative_slope=0.2)
        x_net4l = self.dp2(x_net4l)

        orih_hard = F.leaky_relu(self.bn6(self.linear1(orih_hard)), negative_slope=0.2)
        orih_hard = self.dp1(orih_hard)
        orih_hard = F.leaky_relu(self.bn7(self.linear2(orih_hard)), negative_slope=0.2)
        orih_hard = self.dp2(orih_hard)

        orim_hard = F.leaky_relu(self.bn6(self.linear1(orim_hard)), negative_slope=0.2)
        orim_hard = self.dp1(orim_hard)
        orim_hard = F.leaky_relu(self.bn7(self.linear2(orim_hard)), negative_slope=0.2)
        orim_hard = self.dp2(orim_hard)

        oril_hard = F.leaky_relu(self.bn6(self.linear1(oril_hard)), negative_slope=0.2)
        oril_hard = self.dp1(oril_hard)
        oril_hard = F.leaky_relu(self.bn7(self.linear2(oril_hard)), negative_slope=0.2)
        oril_hard = self.dp2(oril_hard)

        h = x
        # if it needs projection
        if isinstance(self.linear3, ProjHead):
            x = self.linear3(x)
            x_net4h = self.linear3(x_net4h)
            x_net4m = self.linear3(x_net4m)
            x_net4l = self.linear3(x_net4l)

            orih_hard = self.linear3(orih_hard)
            orim_hard = self.linear3(orim_hard)
            oril_hard = self.linear3(oril_hard)

        else:
            x = self.linear3(x)
            x_net4h = self.linear3(x_net4h)
            x_net4m = self.linear3(x_net4m)
            x_net4l = self.linear3(x_net4l)

            orih_hard = self.linear3(orih_hard)
            orim_hard = self.linear3(orim_hard)
            oril_hard = self.linear3(oril_hard)
            
        return h, x, [x_net4h, x_net4m, x_net4l], [orih_hard, orim_hard, oril_hard]


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
    
    def forward(self, pred, gold, trans_feat=None, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss