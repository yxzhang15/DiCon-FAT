# -*- coding: utf-8 -*-

import argparse
from http.client import ImproperConnectionState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
import numpy as np

from model.gcn import TreeGCN

class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x

class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x

class FFTBlock2d(nn.Module):
    def __init__(self, dim):
        super(FFTBlock2d, self).__init__()
        self.frequency_process = frequency_process(dim)
        self.conv = torch.nn.Conv2d(2*dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.spatial = spatial(dim)

    def forward(self,x):
        x_spatial = self.spatial(x)
        x_req = torch.fft.rfft2(x, norm='backward')
        x_req = self.frequency_process(x_req)
        x_back_freq = torch.fft.irfft2(x_req, norm='backward')
        x_out = torch.cat((x_spatial, x_back_freq), dim=1)
        x_out = self.relu(self.bn(self.conv(x_out)))
        return x_out + x

    
class frequency_process(nn.Module):
    def __init__(self, dim):
        super(frequency_process, self).__init__()
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)        
        self.bn2 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.conv1 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True) 
        self.bn1 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
    def forward(self,x):
        mag1 = torch.abs(x)
        pha1 = torch.angle(x)

        mag = self.relu1(self.bn1(self.conv1(mag1)))
        mag = mag1 + mag
        pha = self.relu2(self.bn2(self.conv2(pha1)))
        pha = pha1 + pha

        real_part = mag * torch.cos(pha)
        img_part = mag * torch.sin(pha)
        x = torch.complex(real_part, img_part)
        return x

class spatial(nn.Module):
    def __init__(self, dim):
        super(spatial, self).__init__() 
        self.conv1 = torch.nn.Conv2d(dim, dim, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = torch.nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    def forward(self,x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        return x + x2
        
class Harvester(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Harvester, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)
        self.conv3d = Conv1DBlock((2*in_channel, emb_dims, emb_dims), 1)
        self.fft = FFTBlock2d(3)

    def forward(self, x, idx):
        x_ori = x
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)
        x = nn_feat - x.unsqueeze(-1)
        x = self.conv2d(x)
        x = self.fft(x)
        x = x.max(-1)[0]
        x = self.conv1d(x)
        x = torch.cat((x, x_ori), dim=1)
        x = self.conv3d(x)
        return x
    
class Convlayer(nn.Module):

    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = torch.unsqueeze(x, 1) # [B, 1, N, 3]
        x = F.relu(self.bn1(self.conv1(x))) # [B, 64, N, 1]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 64, N, 1]
        x_128 = F.relu(self.bn3(self.conv3(x))) # [B, 128, N, 1]
        x_256 = F.relu(self.bn4(self.conv4(x_128))) # [B, 256, N, 1]
        x_512 = F.relu(self.bn5(self.conv5(x_256))) # [B, 512, N, 1]
        x_1024 = F.relu(self.bn6(self.conv6(x_512))) # [B, 1024, N, 1]
        x_128 = torch.squeeze(self.maxpool(x_128), 2) # [B, 128, 1]
        x_256 = torch.squeeze(self.maxpool(x_256), 2) # [B, 256, 1]
        x_512 = torch.squeeze(self.maxpool(x_512), 2) # [B, 512, 1]
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2) # [B, 1024, 1]
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1) # [B, 1920, 1]
        return x
    

class Convlayer(nn.Module):

    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = torch.unsqueeze(x, 1) # [B, 1, N, 3]
        x = F.relu(self.bn1(self.conv1(x))) # [B, 64, N, 1]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 64, N, 1]
        x_128 = F.relu(self.bn3(self.conv3(x))) # [B, 128, N, 1]
        x_256 = F.relu(self.bn4(self.conv4(x_128))) # [B, 256, N, 1]
        x_512 = F.relu(self.bn5(self.conv5(x_256))) # [B, 512, N, 1]
        x_1024 = F.relu(self.bn6(self.conv6(x_512))) # [B, 1024, N, 1]
        x_128 = torch.squeeze(self.maxpool(x_128), 2) # [B, 128, 1]
        x_256 = torch.squeeze(self.maxpool(x_256), 2) # [B, 256, 1]
        x_512 = torch.squeeze(self.maxpool(x_512), 2) # [B, 512, 1]
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2) # [B, 1024, 1]
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1) # [B, 1920, 1]
        return x


class Treegcnlayer(nn.Module):

    def __init__(self, args=None):
        super(Treegcnlayer, self).__init__()
        self.args = args
        self.features = self.args.G_FEAT
        self.degrees = self.args.DEGREE
        self.support = self.args.support
        self.loop_non_linear = self.args.loop_non_linear

        self.layer_num = len(self.features) - 1
        assert self.layer_num == len(self.degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None

        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            # NOTE last layer activation False
            if inx == self.layer_num - 1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, self.features, self.degrees, 
                                            support=self.support, node=vertex_num, upsample=True, activation=False, args=self.args))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, self.features, self.degrees, 
                                            support=self.support, node=vertex_num, upsample=True, activation=True, args=self.args))
            vertex_num = int(vertex_num * self.degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)

        self.pointcloud = feat[-1]
        pointcloud = self.pointcloud.transpose(1,2) #b 3 2048
        
        return pointcloud

def knn(x, k):
    # x is of shape [B, N, 3]
    inner = -2 * torch.matmul(x, x.transpose(2, 1).contiguous())
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

class AutoEncoder(nn.Module):

    def __init__(self, k, input_point_nums, decoder_type='normal_conv', args=None):
        super(AutoEncoder, self).__init__()
        self.num_class = k
        self.input_point_nums = input_point_nums
        self.decoder_type = decoder_type
        self.args = args

        ### Label Embedding ###
        # self.embedding = nn.Embedding(self.num_class, self.num_class)
        # self.em_fc1 = nn.Linear(self.num_class, 8)
        # self.em_fc2 = nn.Linear(8, 2)

        ### Encoder ###
        self.ComMLP = Convlayer(point_scales=self.input_point_nums)
        self.fc1 = nn.Linear(1920, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

        ### Decoder ###
        if self.decoder_type == 'normal_conv':
            self.fc1_1 = nn.Linear(1024, 128*64)
            # self.conv1_1 = torch.nn.Conv1d(256, 128, 1)
            self.conv1_2 = torch.nn.Conv1d(64, 64, 1)
            self.conv1_3 = torch.nn.Conv1d(64, int((self.input_point_nums*3)/128), 1)
            self.bn1_1 = nn.BatchNorm1d(128*64)
            self.bn1_2 = nn.BatchNorm1d(64)
            self.bn1_3 = nn.BatchNorm1d(int((self.input_point_nums*3)/128))
        elif self.decoder_type == 'treegcn':
            self.fc2_1 = nn.Linear(1024, 512)
            self.fc2_2 = nn.Linear(512, 256)
            self.fc2_3 = nn.Linear(256, 96)
            self.TreeGCN = Treegcnlayer(args=self.args)
        else:
            assert self.decoder_type in ['normal_conv', 'treegcn'], "Illegal Decoder Type!"
        self.fft = Harvester(3,3)

    def forward(self, x, labels):

        ### Encoder ###
        x = x.transpose(1, 2) # [B, N, 3]
        outs = self.ComMLP(x) # [B, 1920, 1]
        outs = torch.squeeze(outs, 2) # [B, 1920]
        latentfeature = F.relu(self.bn1(self.fc1(outs))) # [B, 1024]

        ### Decoder ###
        if self.decoder_type == 'normal_conv':
            pc_feat = F.relu(self.bn1_1(self.fc1_1(latentfeature))) # [B, 128*64]
            pc_feat = pc_feat.reshape(-1, 64, 128) # [B, 64, 128]
            pc_feat = F.relu(self.bn1_2(self.conv1_2(pc_feat))) # [B, 64, 128]
            pc_xyz = self.bn1_3(self.conv1_3(pc_feat)) # [B, N*3/128, 128]
            pc_xyz = pc_xyz.reshape(-1, 3, self.input_point_nums) # [B, 3, N]
            ### Activation ###
        elif self.decoder_type == 'treegcn':
            pc_feat = F.relu(self.fc2_1(latentfeature)) # [B, 512]
            pc_feat = F.relu(self.fc2_2(pc_feat)) # [B, 256]
            pc_feat = F.relu(self.fc2_3(pc_feat)) # [B, 96]
            pc_feat = torch.unsqueeze(pc_feat, 1)
            pc_feat = [pc_feat]
            pc_xyz = self.TreeGCN(pc_feat)
        else:
            assert self.decoder_type in ['normal_conv', 'treegcn'], "Illegal Decoder Type!"

        ### fre ###
    #    nn_idx = knn(x, k=20)
    #    pc_xyz = self.fft(pc_xyz, nn_idx)
        ### fre ###
        return pc_xyz



class ProjHead(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ProjHead, self).__init__()
        # projection MLP
        self.l1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.l2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.bn1(self.l1(x))
        x = F.relu(x)
        x = self.l2(x)
        return x
        
class ProjHead_deepsu(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ProjHead_deepsu, self).__init__()
        # projection MLP
        self.l1 = nn.Linear(in_dim, in_dim)
        self.l3 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.l2 = nn.Linear(in_dim, in_dim)
        self.l4 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.bn1(self.l3(self.l1(x)))
        x = F.relu(x)
        x = self.bn2(self.l4(self.l2(x)))
        x = F.relu(x)
        return x


class sequeliner(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(sequeliner, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(channel_in, channel_in, kernel_size=kernel_size),
            nn.Conv1d(channel_in, channel_in, kernel_size=1),
            nn.BatchNorm1d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv1d(channel_in, channel_in, kernel_size=kernel_size),
            nn.Conv1d(channel_in, channel_out, kernel_size=1),
            nn.BatchNorm1d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Model Structure of PointCAT.')

    ### data related
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--input_point_nums', type=int, default=2048, help='point nums of each point cloud')

    ### network related
    parser.add_argument('--decoder_type', type=str, default='normal_conv', choices=['normal_conv', 'treegcn'])

    ### TreeGAN architecture related
    parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
    parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
    parser.add_argument('--D_FEAT', type=int, default=[3, 64,  128, 256, 256, 512], nargs='+', help='Features for discriminator.')
    parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
    parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    ### inference test
    input_ = torch.randn(args.batch_size, args.input_point_nums, 3) # [B, N, 3]
    labels = torch.zeros((args.batch_size, 1)).long()
    AE = AutoEncoder(input_point_nums=args.input_point_nums, decoder_type=args.decoder_type, args=args)
    output_ = AE(input_, labels)
    print(AE)
    print(output_)

    input_ = input_.transpose(2, 1) # [B, 3, N]

    ### consistency test
    from classifier import pointnet_cls, pointnet2_cls_msg

    PointNetSimCLR = pointnet_cls.get_model(40, False)
    PointNetSimCLR.eval()
    output_1, output_2 = PointNetSimCLR(input_)

    PointNetSimCLR.train()
    fc_previous = PointNetSimCLR.fc3
    PointNetSimCLR.fc3 = ProjHead(256, 256)
    print(PointNetSimCLR)
    PointNetSimCLR.fc3 = fc_previous

    PointNetSimCLR.eval()
    output_3, output_4 = PointNetSimCLR(input_)
    print(output_1 - output_3)

