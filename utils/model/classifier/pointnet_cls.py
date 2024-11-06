import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model_utils.pointnet_util import PointNetEncoder, feature_transform_reguliarzer
from model.networks import ProjHead

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

    def forward(self, x):
        # if it uses pre-defense
        if self.use_pre_defense:
            assert self.pre_defense_head is not None
            x = self.pre_defense_head(x)

        x, trans, trans_feat, x_net = self.feat(x)
        gf=x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))

        h = x
        # if it needs projection
        if isinstance(self.fc3, ProjHead):
            x = self.fc3(x)
        else:
            x = self.fc3(x)
            x = F.log_softmax(x, dim=1)
       # return h, x,x,x
        return gf, x,x,x


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
