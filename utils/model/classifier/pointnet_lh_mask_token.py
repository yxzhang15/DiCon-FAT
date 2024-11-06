import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model_utils.pointnet_util_dis import PointNetEncoder, feature_transform_reguliarzer
from model.networks import ProjHead, ProjHead_deepsu
from model_utils.FreDe import FreDe, select_top_20_percent_from_all12

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True, use_pre_defense=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

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
        
        orih_hard = select_top_20_percent_from_all12(orih, ori, batch_size)
        oril_hard = select_top_20_percent_from_all12(oril, ori, batch_size)
        orim_hard = select_top_20_percent_from_all12(orim, ori, batch_size)
        
        oril = torch.max(oril, 1, keepdim=True)[0]            
        oril = oril.view(-1, d)
      
        orih = torch.max(orih, 1, keepdim=True)[0]            
        orih = orih.view(-1, d)
        
        orim = torch.max(orim, 1, keepdim=True)[0]            
        orim = orih.view(-1, d)
        return orih, orim, oril, orih_hard, orim_hard, oril_hard  
            

    def forward(self, x):
        ori=x
        if self.use_pre_defense:
            assert self.pre_defense_head is not None 
            x = self.pre_defense_head(x) 

        x, trans, trans_feat, x_net = self.feat(x)
        num_points = ori.size(-1)
        xh, xm, xl = FreDe(ori, num_points // 3)
        x_net = x_net[-1]

        x_net4h, x_net4m, x_net4l, orih_hard, orim_hard, oril_hard = self.trans_multi(x_net, xh, xl, xm, num_points // 3, 1024)

        
        x_net4h = F.relu(self.bn1(self.fc1(x_net4h))) #512
        x_net4h = F.relu(self.bn2(self.dropout(self.fc2(x_net4h))))        

        x_net4l = F.relu(self.bn1(self.fc1(x_net4l))) #512
        x_net4l = F.relu(self.bn2(self.dropout(self.fc2(x_net4l))))
        
        x_net4m = F.relu(self.bn1(self.fc1(x_net4m))) #512
        x_net4m = F.relu(self.bn2(self.dropout(self.fc2(x_net4m))))
        
        orih_hard = F.relu(self.bn1(self.fc1(orih_hard))) #512
        orih_hard = F.relu(self.bn2(self.dropout(self.fc2(orih_hard))))        

        orim_hard = F.relu(self.bn1(self.fc1(orim_hard))) #512
        orim_hard = F.relu(self.bn2(self.dropout(self.fc2(orim_hard))))
        
        oril_hard = F.relu(self.bn1(self.fc1(oril_hard))) #512
        oril_hard = F.relu(self.bn2(self.dropout(self.fc2(oril_hard))))

        x = F.relu(self.bn1(self.fc1(x))) #512
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))

        h = x
        # if it needs projection
        if isinstance(self.fc3, ProjHead_deepsu):
            x = self.fc3(x)
          #  print('projx', x.size())
            x_net4h = self.fc3(x_net4h)
            x_net4l = self.fc3(x_net4l)
            x_net4m = self.fc3(x_net4m)
            
            orih_hard = self.fc3(orih_hard)
            orim_hard = self.fc3(orim_hard)
            oril_hard = self.fc3(oril_hard)
        else:
            x = self.fc3(x)
            x = F.log_softmax(x, dim=1)
            x_net4h = self.fc3(x_net4h)
            x_net4l = self.fc3(x_net4l)
            x_net4m = self.fc3(x_net4m)
            
            orih_hard = self.fc3(orih_hard)
            orim_hard = self.fc3(orim_hard)
            oril_hard = self.fc3(oril_hard)
            
            x_net4h = F.log_softmax(x_net4h, dim=1)
            x_net4l = F.log_softmax(x_net4l, dim=1)
            x_net4m = F.log_softmax(x_net4m, dim=1)
            
            orih_hard = F.log_softmax(orih_hard, dim=1)
            orim_hard = F.log_softmax(orim_hard, dim=1)
            oril_hard = F.log_softmax(oril_hard, dim=1)

        return h, x, [x_net4h, x_net4m, x_net4l], [orih_hard, orim_hard, oril_hard]
        

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
