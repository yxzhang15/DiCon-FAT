# -*- coding: utf-8 -*-

import os
import time
import random
import importlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils

### arguments related
from arguments import Arguments

### data related
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader, TensorDataset
import utils.provider as provider

### model related
from model_diconfat import DiCon_FAT
from loss import *

### others related ###
from utils.logging import Logging_str
from utils.utils import set_seed
torch.set_num_threads(2)


class Trainer(object):

    def __init__(self, args):
        self.args = args

        ### Check and Create Log Path ###
        log_path = Path('./log/')
        log_path.mkdir(exist_ok=True)
        log_path = log_path.joinpath(self.args.experiment_dir)
        log_path.mkdir(exist_ok=True)
        self.log_path = log_path
        self.logfile_path = os.path.join(self.log_path, 'log_info.txt')
        self.log_string = Logging_str(self.logfile_path)

        ### Model Function Initialization ###
        self.model = DiCon_FAT(self.args)

        ### Dataset Preparation ###
        self.load_data()


    def load_data(self):
        """
        Load the data file. When running our code in the first time, we 
        intend to save .pt data file to acelerate the training procedure.
        """
        self.log_string.write('++++++++++Begin Load Data++++++++++')

        # make directory
        data_pt_path = './data/' + str(self.args.dataset) + '_' + str(self.args.input_point_nums) + '_pt/'
        if not os.path.isdir(data_pt_path):
            os.mkdir(data_pt_path)

        # make data file if not existing
        pt_path_train_points = os.path.join(data_pt_path, 'train_points.pt')
        pt_path_train_target = os.path.join(data_pt_path, 'train_target.pt')
        pt_path_test_points = os.path.join(data_pt_path, 'test_points.pt')
        pt_path_test_target = os.path.join(data_pt_path, 'test_target.pt')
        if not os.path.isfile(pt_path_train_points) or not os.path.isfile(pt_path_train_target) \
            or not os.path.isfile(pt_path_test_points) or not os.path.isfile(pt_path_test_target):
            self.make_dataloader()
            self.save_data_as_pt(self.trainDataLoader, data_pt_path, mode='train')
            self.save_data_as_pt(self.testDataLoader, data_pt_path, mode='test')

        # load the pt file
        points_train = torch.load(pt_path_train_points) # [M_tr, N, C]
        target_train = torch.load(pt_path_train_target) # [M_tr, 1]
        points_test = torch.load(pt_path_test_points) # [M_te, N, C]
        target_test = torch.load(pt_path_test_target) # [M_te, 1]

        # set tensor loader
        self.make_dataloader(
            trainset=tuple([points_train, target_train]),
            testset=tuple([points_test, target_test])
        )

        self.log_string.write('++++++++++End Load Data++++++++++')


    def make_dataloader(self, trainset=None, testset=None):
        """
        Load the dataloader for training or testing.
        """
        if trainset != None and testset != None:
            TRAIN_DATASET = TensorDataset(trainset[0], trainset[1])
            TEST_DATASET = TensorDataset(testset[0], testset[1])
        else:
            if self.args.dataset == 'ModelNet40':
                TRAIN_DATASET = ModelNetDataLoader(
                    root=self.args.data_path, 
                    npoint=self.args.input_point_nums, 
                    split='train', 
                    normal_channel=self.args.normal
                )
                TEST_DATASET = ModelNetDataLoader(
                    root=self.args.data_path, 
                    npoint=self.args.input_point_nums, 
                    split='test', 
                    normal_channel=self.args.normal
                )
            elif self.args.dataset == 'ShapeNetPart':
                TRAIN_DATASET = PartNormalDataset(
                    root=self.args.data_path, 
                    npoints=self.args.input_point_nums, 
                    split='trainval', 
                    normal_channel=self.args.normal
                )
                TEST_DATASET = PartNormalDataset(
                    root=self.args.data_path, 
                    npoints=self.args.input_point_nums, 
                    split='test', 
                    normal_channel=self.args.normal
                )
            else:
                raise NotImplementedError

        self.trainDataLoader = torch.utils.data.DataLoader(
            TRAIN_DATASET, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # pin_memory=True,
            num_workers=self.args.num_workers,
            drop_last=True
        )
        self.testDataLoader = torch.utils.data.DataLoader(
            TEST_DATASET, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            # pin_memory=True,
            num_workers=self.args.num_workers
        )
        self.testDataLoader_forshow = torch.utils.data.DataLoader(
            TEST_DATASET, 
            batch_size=self.args.test_batch_size, #dgcnn 112
           # batch_size=88,# pointnet2
            shuffle=False, 
            # pin_memory=True,
            num_workers=self.args.num_workers
        )


    def save_data_as_pt(self, dataloader, pt_path, mode='train'):
        """
        Turn the given dataloader to data tensors and save them.
        """
        for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9):
            # prepare data for training
            points, target = data[:2]
            # concat to a big tensor
            if batch_id == 0:
                points_gather = points
                target_gather = target
            else:
                points_gather = torch.cat([points_gather, points], dim=0)
                target_gather = torch.cat([target_gather, target], dim=0)
        # get the path to save
        pt_path = pt_path + str(mode)
        points_path = pt_path + '_points.pt'
        target_path = pt_path + '_target.pt'
        # save the pt
        torch.save(points_gather, points_path)
        torch.save(target_gather, target_path)


    def data_preprocess(self, data, mode='train'):
        """
        Preprocess the given data and label.
        """
        points, target = data

        if mode == 'train':
            # pointcloud augmentation
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points) # [B, N, C]

        points = points.transpose(2, 1) # [B, C, N]
        target = target[:, 0] # [B]

        points = points.cuda()
        target = target.cuda()

        return points, target

    def run(self):
        """
        The framework support the following modes:
        - 
        """
        if True:
            self.train()
        else:
            raise NotImplementedError


    def train(self):
        """
        Implement the training procedure.
        """

        # preparation
        self.model.build_optimizers()
        self.model.set_loss_function()
        self.model.get_feature_peak(mode='init')
        if self.args.defended_model == 'pointnet_lh_mask_token':
            best_acc = 0.87 if self.args.dataset == 'ModelNet40' else 0.95
        else:
            best_acc = 0.91 if self.args.dataset == 'ModelNet40' else 0.95

        # start DiCon_FAT training
        self.log_string.write('++++++++++Begin DiCon_FAT Training++++++++++')
        for epoch in range(1, self.args.epochs+1):

            if epoch == 1:
                # save fixed fc, initialize projection head
                self.model.get_projection_head(mode='add')
            else:
                # save fixed fc, use trained projection head
                self.model.get_projection_head(mode='reverse')

            # get overall epoch nums
            epoch_c = self.model.start_epoch_c + epoch
            epoch_ng = self.model.start_epoch_ng + epoch
            self.log_string.write('Epoch_c %d, Epoch_ng %d, (%d/%s):' % (epoch_c, epoch_ng, epoch, self.args.epochs))

            # start contrastive adversarial training
            for batch_id, data in tqdm(enumerate(self.trainDataLoader), total=len(self.trainDataLoader)):
                # prepare data for training
                points, target = self.data_preprocess(data, mode='train')
                self.model.set_target(pc=points, target=target)

                # train on the batch data
                self.model.run(mode='train')

                # dynamic fp update
                self.model.get_projection_head(mode='reverse')
                self.model.get_feature_peak(mode='update')
                self.model.get_projection_head(mode='reverse')

            # change head for validation
            # save trained projection head, use fixed fc
            self.model.get_projection_head(mode='reverse')

            # start fine-tuning
            for batch_id, data in tqdm(enumerate(self.trainDataLoader), total=len(self.trainDataLoader)):
                # prepare data for training
                points, target = self.data_preprocess(data, mode='train')
                self.model.set_target(pc=points, target=target)

                # train on the batch data
                signal = True if epoch == 1 and batch_id == 0 else False
                self.model.run(mode='finetune', ii=signal)

            # update learning rate schedulers
            self.model.scheduler_step()

            # start clean acc validation
            if epoch % 1 == 0:
                acc_num = 0
                all_num = 0
                self.model.classifier.eval()
                for batch_id, data in tqdm(enumerate(self.testDataLoader), total=len(self.testDataLoader)):
                    # prepare data for testing
                    points, target = self.data_preprocess(data, mode='test')
                    points = points
                    target = target.long()

                    # test on the batch data
                    with torch.no_grad():
                        _, logits,x1, x2 = self.model.classifier(points)
                        pred = torch.argmax(logits, dim=-1)
                        acc_num += (pred == target).sum().item()
                        all_num += target.size(0)
                new_acc = acc_num / float(all_num)
                print('++++++++++Begin Test++++++++++')
                self.log_string.write('Current Acc: %.6f' %(new_acc))

            # get the best point
            if new_acc > best_acc:
                # start robustness validation
                self.log_string.write('++++++++++Begin Validation++++++++++')
                self.model.show_results(mode='init')
                for batch_id, data in tqdm(enumerate(self.testDataLoader_forshow), total=len(self.testDataLoader_forshow)):
                    # prepare data for testing
                    points, target = self.data_preprocess(data, mode='test')
                    self.model.set_target(pc=points, target=target)

                    # test on the batch data
                    self.model.run(mode='test')

                # show performance
                self.model.show_results(mode='print')

                # save the best model
                best_acc = self.model.acc_clean.avg
                
                self.log_string.write('++++++++++Begin Saving Model++++++++++')
                save_path = self.log_path.joinpath('./checkpoints/')
                save_path.mkdir(exist_ok=True)
                self.model.save_checkpoints(save_path, epoch_c, epoch_ng, mode='best')
                self.log_string.write('++++++++++End Saving Model++++++++++')


        # save checkpoints
        self.log_string.write('++++++++++Begin Saving Latest Model++++++++++')
        save_path = self.log_path.joinpath('./checkpoints/')
        save_path.mkdir(exist_ok=True)
        self.model.save_checkpoints(save_path, epoch_c, epoch_ng, mode='latest')
        self.log_string.write('++++++++++End Saving Latest Model++++++++++')



if __name__ == "__main__":
    args = Arguments(stage='train').parser().parse_args()
    # args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #  print(args)
    
    if not os.path.isdir('./log/'):
        os.mkdir('./log/')
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    set_seed(2022)
    trainer = Trainer(args)
    trainer.run()
    
