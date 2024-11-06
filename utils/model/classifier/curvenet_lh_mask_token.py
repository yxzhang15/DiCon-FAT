"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_cls.py
@Time: 2021/01/21 3:10 PM
"""

import torch.nn as nn
import torch.nn.functional as F
from model_utils.curvenet_util import *
from model.networks import ProjHead
from model_utils.FreDe import FreDe, select_top_20_percent_from_all34

curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }
    
class get_model(nn.Module):
    def __init__(self, num_classes=40, normal_channel=False, k=20, setting='default', use_pre_defense=False):
        super(get_model, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

        self.use_pre_defense = use_pre_defense
        
    def trans_multi(self, ori, indices, indicesl, indicesm, M, d, mask_ratio=None):
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
        orim_hard = select_top_20_percent_from_all34(orim, ori, batch_size)
        oril_hard = select_top_20_percent_from_all34(oril, ori, batch_size)
        
        return orih, oril, orim, orih_hard, orim_hard, oril_hard


    def set_pre_head(self, pre_defense_head):
        # if we need preprocess-based defense module
        assert pre_defense_head is not None
        self.pre_defense_head = pre_defense_head


    def forward(self, xyz, mask_ratio=None):
        # if it uses pre-defense
        if self.use_pre_defense:
            assert self.pre_defense_head is not None
            xyz = self.pre_defense_head(xyz)
        batch_size = xyz.size(0)
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)
        
        num_points = l4_xyz.size(-1)
        xh, xm, xl = FreDe(l4_xyz, num_points // 3)
        
        x_net4h, x_net4l, x_net4m, orih_hard, orim_hard, oril_hard = self.trans_multi(l4_points, xh, xl, xm, num_points // 3, 1024)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)   
        gf = x_avg.squeeze(-1)     
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        
        x_net4h = self.conv0(x_net4h.permute(0,2,1))
        x1h = F.adaptive_max_pool1d(x_net4h, 1).view(batch_size, -1)
        x2h = F.adaptive_avg_pool1d(x_net4h, 1).view(batch_size, -1)
        x_net4h = torch.cat((x1h, x2h), 1)
        
        x_net4m = self.conv0(x_net4m.permute(0,2,1))
        x1m = F.adaptive_max_pool1d(x_net4m, 1).view(batch_size, -1)
        x2m = F.adaptive_avg_pool1d(x_net4m, 1).view(batch_size, -1)
        x_net4m = torch.cat((x1m, x2m), 1)
        
        x_net4l = self.conv0(x_net4l.permute(0,2,1))
        x1l = F.adaptive_max_pool1d(x_net4l, 1).view(batch_size, -1)
        x2l = F.adaptive_avg_pool1d(x_net4l, 1).view(batch_size, -1)
        x_net4l = torch.cat((x1l, x2l), 1)

        orih_hard = self.conv0(orih_hard.permute(0,2,1))
        x1h = F.adaptive_max_pool1d(orih_hard, 1).view(batch_size, -1)
        x2h = F.adaptive_avg_pool1d(orih_hard, 1).view(batch_size, -1)
        orih_hard = torch.cat((x1h, x2h), 1)

        orim_hard = self.conv0(orim_hard.permute(0,2,1))
        x1h = F.adaptive_max_pool1d(orim_hard, 1).view(batch_size, -1)
        x2h = F.adaptive_avg_pool1d(orim_hard, 1).view(batch_size, -1)
        orim_hard = torch.cat((x1h, x2h), 1)

        oril_hard = self.conv0(oril_hard.permute(0,2,1))
        x1h = F.adaptive_max_pool1d(oril_hard, 1).view(batch_size, -1)
        x2h = F.adaptive_avg_pool1d(oril_hard, 1).view(batch_size, -1)
        oril_hard = torch.cat((x1h, x2h), 1)
        
        x_net4h = F.relu(self.bn1(self.conv1(x_net4h).unsqueeze(-1)), inplace=True).squeeze(-1)
        x_net4h = self.dp1(x_net4h)
        x_net4m = F.relu(self.bn1(self.conv1(x_net4m).unsqueeze(-1)), inplace=True).squeeze(-1)
        x_net4m = self.dp1(x_net4m) 
        x_net4l = F.relu(self.bn1(self.conv1(x_net4l).unsqueeze(-1)), inplace=True).squeeze(-1)
        x_net4l = self.dp1(x_net4l)

        orih_hard = F.relu(self.bn1(self.conv1(orih_hard).unsqueeze(-1)), inplace=True).squeeze(-1)
        orih_hard = self.dp1(orih_hard)
        orim_hard = F.relu(self.bn1(self.conv1(orim_hard).unsqueeze(-1)), inplace=True).squeeze(-1)
        orim_hard = self.dp1(orim_hard) 
        oril_hard = F.relu(self.bn1(self.conv1(oril_hard).unsqueeze(-1)), inplace=True).squeeze(-1)
        oril_hard = self.dp1(oril_hard)
        
        h = x
        # if it needs projection
        if isinstance(self.conv2, ProjHead):
            x = self.conv2(x)
            x_net4h = self.conv2(x_net4h)
            x_net4m = self.conv2(x_net4m)
            x_net4l = self.conv2(x_net4l)
            orih_hard = self.conv2(orih_hard)
            orim_hard = self.conv2(orim_hard)
            oril_hard = self.conv2(oril_hard)
        else:
            x = self.conv2(x)
            x_net4h = self.conv2(x_net4h)
            x_net4m = self.conv2(x_net4m)
            x_net4l = self.conv2(x_net4l)
            orih_hard = self.conv2(orih_hard)
            orim_hard = self.conv2(orim_hard)
            oril_hard = self.conv2(oril_hard)
        return gf, x, [x_net4h, x_net4m, x_net4l], [orih_hard, orim_hard, oril_hard]
