import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils.pointnet2_util import PointNetSetAbstractionMsg, PointNetSetAbstraction_formask
from model.networks import ProjHead
from model_utils.FreDe import FreDe, select_top_20_percent_from_all12

class get_model(nn.Module):
    def __init__(self,k=40,normal_channel=True, use_pre_defense=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction_formask(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, k)

        self.use_pre_defense = use_pre_defense

    def set_pre_head(self, pre_defense_head):
        # if we need preprocess-based defense module
        assert pre_defense_head is not None
        self.pre_defense_head = pre_defense_head
        
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
        
        orih_hard = select_top_20_percent_from_all12(orih, ori, batch_size)
        oril_hard = select_top_20_percent_from_all12(oril, ori, batch_size)
        orim_hard = select_top_20_percent_from_all12(orim, ori, batch_size)

        orih = torch.max(orih, 1)[0]            
        orih = orih.view(-1, d)
        
        orim = torch.max(orim, 1)[0]            
        orim = orim.view(-1, d)

        oril = torch.max(oril, 1)[0]            
        oril = oril.view(-1, d)
        return orih, oril, orim, orih_hard, orim_hard, oril_hard

    def forward(self, xyz, mask_ratio=None):
        # if it uses pre-defense
        if self.use_pre_defense:
            assert self.pre_defense_head is not None
            xyz = self.pre_defense_head(xyz)

        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points_pool, l3_points = self.sa3(l2_xyz, l2_points)
     #   print('l2_points', l2_points.size())
        num_points = l2_xyz.size(-1)
        xh, xm, xl = FreDe(l2_xyz, num_points // 3)
        
        if mask_ratio:
            x_net4h, x_net4l, x_net4m, orih_hard, orim_hard, oril_hard = self.trans_multi(l3_points, xh, xl, xm, num_points // 3, 1024, mask_ratio)
        else:
            x_net4h, x_net4l, x_net4m, orih_hard, orim_hard, oril_hard = self.trans_multi(l3_points, xh, xl, xm, num_points // 3, 1024)
            
        x = l3_points_pool.view(B, 1024)
        gf=x
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        x_net4h = self.drop1(F.relu(self.bn1(self.fc1(x_net4h))))
        x_net4h = self.drop2(F.relu(self.bn2(self.fc2(x_net4h))))
        x_net4l = self.drop1(F.relu(self.bn1(self.fc1(x_net4l))))
        x_net4l = self.drop2(F.relu(self.bn2(self.fc2(x_net4l))))
        x_net4m = self.drop1(F.relu(self.bn1(self.fc1(x_net4m))))
        x_net4m = self.drop2(F.relu(self.bn2(self.fc2(x_net4m))))
        
        orih_hard = self.drop1(F.relu(self.bn1(self.fc1(orih_hard))))
        orih_hard = self.drop2(F.relu(self.bn2(self.fc2(orih_hard))))
        orim_hard = self.drop1(F.relu(self.bn1(self.fc1(orim_hard))))
        orim_hard = self.drop2(F.relu(self.bn2(self.fc2(orim_hard))))
        oril_hard = self.drop1(F.relu(self.bn1(self.fc1(oril_hard))))
        oril_hard = self.drop2(F.relu(self.bn2(self.fc2(oril_hard))))

        h = x
        # if it needs projection
        if isinstance(self.fc3, ProjHead):
            x = self.fc3(x)
            x_net4h = self.fc3(x_net4h)
            x_net4l = self.fc3(x_net4l)
            x_net4m = self.fc3(x_net4m)

            orih_hard = self.fc3(orih_hard)
            orim_hard = self.fc3(orim_hard)
            oril_hard = self.fc3(oril_hard)
        else:
            x = self.fc3(x)
            x_net4h = self.fc3(x_net4h)
            x_net4l = self.fc3(x_net4l)
            x_net4m = self.fc3(x_net4m)
            x = F.log_softmax(x, -1)
            x_net4h = F.log_softmax(x_net4h, -1)
            x_net4l = F.log_softmax(x_net4l, -1)
            x_net4m = F.log_softmax(x_net4m, -1)   

            orih_hard = self.fc3(orih_hard)
            orim_hard = self.fc3(orim_hard)
            oril_hard = self.fc3(oril_hard)
            orih_hard = F.log_softmax(orih_hard, -1)
            orim_hard = F.log_softmax(orim_hard, -1)
            oril_hard = F.log_softmax(oril_hard, -1)   


        return h, x, [x_net4h, x_net4m, x_net4l], [orih_hard, orim_hard, oril_hard]


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


