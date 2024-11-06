# -*- coding: utf-8 -*-

import os
import sys
import datetime
# import logging
import shutil
import importlib
import numpy as np
from pathlib import Path

import torch
import torch.optim
import torchvision.utils as vutils
from torch.autograd import Variable

from tqdm import tqdm
import utils.provider as provider


from utils.logging import Logging_str
from utils.utils import AverageMeter, save_checkpoint
from model.networks import AutoEncoder, ProjHead_deepsu #, Generator, Discriminator
from loss.nt_adv import NTAdvLoss
from loss.ICL import ICL1, ICL2
from loss.GCL import GCL
from baselines import *


# include other paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))
torch.set_num_threads(2)

class DiCon_FAT(object):
    def __init__(self, args):
        self.args = args

        ### Create Log Path ###
        self.log_path = os.path.join('./log/', self.args.experiment_dir)
        self.logfile_path = os.path.join(self.log_path, 'log_info.txt')
        self.log_string = Logging_str(self.logfile_path)

        ### Initialize Settings ###
        self.need_preview = False

        if self.args.dataset == 'ModelNet40':
            self.num_class = 40
        elif self.args.dataset == 'ShapeNetPart':
            self.num_class = 16
        else:
            raise NotImplementedError

        ### Initialize Model and Weights ###
        self.build_models()
        self.load_weights()



    def build_models(self):
        """
        Build new models for training.
        """
        MODEL = importlib.import_module(self.args.defended_model)
        classifier = MODEL.get_model(
            self.num_class, 
            normal_channel=self.args.normal
        )
        noise_generator = AutoEncoder(
            k=self.num_class,
            input_point_nums=self.args.input_point_nums, 
            decoder_type=self.args.decoder_type, 
            args=self.args
        )

        # use multiple gpus
        if self.args.use_multi_gpu:
            classifier = nn.DataParallel(classifier)
            noise_generator = nn.DataParallel(noise_generator)
        
        self.classifier = classifier.cuda()
        self.noise_generator = noise_generator.cuda()


    def load_weights(self):
        """
        Load weights from checkpoints.
        """
        try:
            checkpoint = torch.load(str(self.log_path) + '/checkpoints/classifier_noAT.pth')
            # checkpoint = torch.load('../Pointnet_Pointnet2_pytorch/log/classification_old/' \
            #     + self.args.defended_model + '/checkpoints/best_model.pth')
            self.start_epoch_c = checkpoint['epoch']
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.log_string.write('Use pretrained classifier')
        except:
            self.start_epoch_c = 0
            # self.need_preview = True
            self.log_string.write('++++++++++Begin training from scratch++++++++++')
        
        try:
            checkpoint = torch.load(str(self.log_path) + '/checkpoints/best_noise_generator.pth')
            self.start_epoch_ng = checkpoint['epoch']
            self.noise_generator.load_state_dict(checkpoint['model_state_dict'])
            self.log_string.write('Use pretrained noise-generator')
        except:
            self.start_epoch_ng = 0
            self.log_string.write('++++++++++Begin training from scratch++++++++++')


    def build_optimizers(self):
        """
        Build new optimizers for each model respectively.
        """
        if self.args.optimizer == 'Adam':
            self.optimizer_c = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.args.lr_c,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.args.decay_rate
            )
            self.optimizer_ng = torch.optim.Adam(
                self.noise_generator.parameters(),
                lr=self.args.lr_ng,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.args.decay_rate
            )
        else:
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(), 
                lr=self.args.lr_c, 
                momentum=0.9
            )
            self.optimizer_ng = torch.optim.SGD(
                self.noise_generator.parameters(), 
                lr=self.args.lr_ng, 
                momentum=0.9
            )
        # self.scheduler_c = torch.optim.lr_scheduler.StepLR(self.optimizer_c, step_size=20, gamma=0.7)
        # self.scheduler_ng = torch.optim.lr_scheduler.StepLR(self.optimizer_ng, step_size=20, gamma=0.7)
        self.scheduler_c = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_c,
            T_0=5,
            T_mult=2
        )
        self.scheduler_ng = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_ng,
            T_0=5,
            T_mult=2
        )     


    def scheduler_step(self):
        """
        Update learning rate schedulers.
        """
        self.scheduler_c.step()
        self.scheduler_ng.step()
        self.scheduler_fc.step()


    def clear_grad(self, model):
        """
        Clear gradient buffers of model.
        """
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    
    
    def reset_grad(self):
        """
        Reset the gradient buffers.
        """
        self.optimizer_c.zero_grad()
        self.optimizer_ng.zero_grad()



    def get_projection_head(self, mode):
        """
        Support the following changing mode:
        - add: replace the final layer with a new projection head, and save the previous final layer.
        - reverse: replace the final layer with the saved previous layer.
        """
        if mode == 'add':
            self.fc_previous = None
            if self.args.use_multi_gpu:
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    ch = self.classifier.module.linear3.in_features
                    self.fc_previous = self.classifier.module.linear3
                    self.classifier.module.linear3 = ProjHead_deepsu(in_dim=ch, out_dim=ch)
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    ch = self.classifier.module.conv2.in_features
                    self.fc_previous = self.classifier.module.conv2
                    self.classifier.module.conv2 = ProjHead_deepsu(in_dim=ch, out_dim=ch)
                else:
                    ch = self.classifier.module.fc3.in_features
                    self.fc_previous = self.classifier.module.fc3
                    self.classifier.module.fc3 = ProjHead_deepsu(in_dim=ch, out_dim=ch)
            else:
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    ch = self.classifier.linear3.in_features
                    self.fc_previous = self.classifier.linear3
                    self.classifier.linear3 = ProjHead_deepsu(in_dim=ch, out_dim=ch)
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    ch = self.classifier.conv2.in_features
                    self.fc_previous = self.classifier.conv2
                    self.classifier.conv2 = ProjHead_deepsu(in_dim=ch, out_dim=ch)
                else:
                    ch = self.classifier.fc3.in_features
                    self.fc_previous = self.classifier.fc3
                    self.classifier.fc3 = ProjHead_deepsu(in_dim=ch, out_dim=ch)

        elif mode == 'reverse':
            if self.args.use_multi_gpu:
                assert self.fc_previous is not None, "No existing previous final-layer !!!"
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    temp_head = self.classifier.module.linear3
                    self.classifier.module.linear3 = self.fc_previous
                    self.fc_previous = temp_head
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    temp_head = self.classifier.module.conv2
                    self.classifier.module.conv2 = self.fc_previous
                    self.fc_previous = temp_head
                else:
                    temp_head = self.classifier.module.fc3
                    self.classifier.module.fc3 = self.fc_previous
                    self.fc_previous = temp_head
            else:
                assert self.fc_previous is not None, "No existing previous final-layer !!!"
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    temp_head = self.classifier.linear3
                    self.classifier.linear3 = self.fc_previous
                    self.fc_previous = temp_head
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    temp_head = self.classifier.conv2
                    self.classifier.conv2 = self.fc_previous
                    self.fc_previous = temp_head
                else:
                    temp_head = self.classifier.fc3
                    self.classifier.fc3 = self.fc_previous
                    self.fc_previous = temp_head

        else:
            raise NotImplementedError

        self.classifier = self.classifier.cuda()


    def CWLoss(self, logits, target, kappa=0, tar=True, num_classes=40):
        """
        C&W loss function.
        """
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        target_one_hot = Variable(torch.eye(num_classes).type(torch.cuda.FloatTensor)[target.long()].cuda())
        
        real = torch.sum(target_one_hot*logits, 1)
        other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
        kappa = torch.zeros_like(other).fill_(kappa)

        if tar:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))


    def get_feature_peak(self, mode='init'):
        self.classifier.eval()
        feature_gather = None
        input_gather = None
        ### Per-class Feature Centre Optimization
        # label initializing
        labels = torch.Tensor([y for y in range(self.num_class)]).cuda() # [num_class]

        # input initializing
        if mode == 'init':
            if self.args.normal:
                random_input = 2 * torch.rand(self.num_class, 6, self.args.input_point_nums) - 1
            else:
                random_input = 2 * torch.rand(self.num_class, 3, self.args.input_point_nums) - 1
        else:
            random_input = self.input_gather

        random_input = random_input.clone().detach_()
        random_input = Variable(random_input.cuda().float()) # [num_class, C, N]
        random_input.requires_grad = True

        # optimizer initializing
        lr = 0.005 if mode == 'init' else self.args.lr_fp
        optimizer = torch.optim.Adam(
            params=[random_input],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )

        # define loop nums
        loop_nums = self.args.init_search_iters if mode == 'init' else self.args.update_search_iters

        # optimizing loop
        for _ in range(1, loop_nums):
            optimizer.zero_grad()

            input = torch.clamp(random_input, min=-1, max=1)
            _, pred,x_x1,x_l2 = self.classifier(input) # [num_class, num_class, 1]
            self.classifier.zero_grad()

            # mask = torch.ones_like(pred[0]).float().cuda()
            # mask[y.long().data] = - 999999 * pred[0][y.long().data].sign()
            # confidence = pred[0][y.long().data] - torch.max(mask * pred[0])
            score = self.CWLoss(pred, labels, kappa=-100, tar=True, num_classes=self.num_class)
            score.backward()

            optimizer.step()

        # get representaions
        with torch.no_grad():
            rep, pred,pro,x_l2= self.classifier(random_input)

        feature_gather = rep.data
        input_gather = random_input.data
        pro_net1 = pro[0].data
        pro_net2 = pro[1].data
        pro_net3 = pro[2].data        

        ### Save to Constant
        self.feature_gather = feature_gather.clone().detach_()
        self.input_gather = input_gather.clone().detach_()
        self.pro_net3 = pro_net3.clone().detach_()
        self.pro_net1 = pro_net1.clone().detach_()
        self.pro_net2 = pro_net2.clone().detach_()

    def set_target(self, pc, target):
        """
        Set labels and original examples from dataset.
        """
        self.pc_ori = pc # [B, C, N]
        self.target = target.long() # [B]


    def set_loss_function(self):
        """
        Set loss functions for training.
        """
        ### Initialize Loss Function ###
        self.GCL = GCL(
            device=self.args.device,
            batch_size=self.args.batch_size,
            temperature=self.args.tm_gcl,
            use_cosine_similarity=self.args.use_cosine_similarity
        )
        self.ICL1 = ICL1(
            temperature=self.args.tm_icl1,
            use_cosine_similarity=self.args.use_cosine_similarity
        )
        self.ICL2 = ICL2(
            device=self.args.device,
            batch_size=self.args.batch_size,
            temperature=self.args.tm_icl2,
            use_cosine_similarity=self.args.use_cosine_similarity
        )
        self.nt_adv_criterion = NTAdvLoss(
            device=self.args.device,
            batch_size=self.args.batch_size,
            beta=self.args.beta,
            temperature=self.args.tm_adv,
            use_cosine_similarity=self.args.use_cosine_similarity
        )        


    def run(self, mode, ii=False):
        """
        Support the following running mode:
        - training: implement CAT on the prepared classifier.
        - testing: test the accuracy and the robustness of CAT trained classifiers.
        """
        if mode == 'train':
            self.run_train()
        elif mode == 'test':
            self.run_test()
        elif mode == 'finetune':
            self.run_finetune(ii)
        else:
            raise NotImplementedError

    def run_train(self):
        """
        Implement CAT on the prepared classifier.
        """
        self.classifier.eval()
        self.noise_generator.train()

        for _ in range(self.args.inner_loop_nums):

            # Compute adversarial perturbation
            pert = self.noise_generator(self.pc_ori, self.target)
            norm = torch.sum(pert ** 2, dim=[1, 2]) ** 0.5
            pert = pert / (norm[:, None, None] + 1e-9)
            pert = pert * np.sqrt(self.pc_ori.size(1) * self.pc_ori.size(2))
            pert = pert * self.args.eps


            # Compute adversarial point-cloud
            self.pc_adv = torch.clamp(self.pc_ori+pert, min=-1, max=1)

            # Get two projections respectively
            _, projection_ori,ori_l1,ori_l2 = self.classifier(self.pc_ori) # [B, ch]
            _, projection_adv,adv_l1,adv_l2 = self.classifier(self.pc_adv) # [B, ch]

            # Get intra-class feature centre for current batch data
            feature_peak = self.feature_gather.index_select(0, self.target) # [B, 1, ch]
            # feature_peak = feature_peak.squeeze(1) # [B, ch]

            # Get the projection of feature peak for the batch data
            projection_fp = None
            if self.args.use_multi_gpu:
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    projection_fp = self.classifier.module.linear3(feature_peak) # [B, ch]
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    projection_fp = self.classifier.module.conv2(feature_peak) # [B, ch]
                else:
                    projection_fp = self.classifier.module.fc3(feature_peak) # [B, ch]
            else:
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    projection_fp = self.classifier.linear3(feature_peak) # [B, ch]
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    projection_fp = self.classifier.conv2(feature_peak) # [B, ch]
                else:                
                    projection_fp = self.classifier.fc3(feature_peak) # [B, ch]
            assert projection_fp is not None

            # Compute the adversarial loss and update noise-generator
            loss_ng = self.nt_adv_criterion(ori=projection_ori, adv=projection_adv, fp=projection_fp)
            self.optimizer_ng.zero_grad()
            loss_ng.backward()
            self.optimizer_ng.step()

        self.classifier.train()
        self.noise_generator.eval()

        del pert, self.pc_adv, projection_adv, projection_fp, feature_peak

        # Compute adversarial perturbation
        pert = self.noise_generator(self.pc_ori, self.target)
        norm = torch.sum(pert ** 2, dim=[1, 2]) ** 0.5
        pert = pert / (norm[:, None, None] + 1e-9)
        pert = pert * np.sqrt(self.pc_ori.size(1) * self.pc_ori.size(2))
        pert = pert * self.args.eps

        # Compute adversarial point-cloud
        self.pc_adv = torch.clamp(self.pc_ori+pert, min=-1, max=1)

        # Re-get the projection of adv
        feature_ori, projection_ori, ori_net, ori_net_hard = self.classifier(self.pc_ori)
        feature_adv, projection_adv, adv_net, adv_net_hard = self.classifier(self.pc_adv.detach()) # [B, ch]

        feature_peak = self.feature_gather # [M, ch]
              
        # Get the projection of feature peak for the batch data
        projection_fp = None
        if self.args.use_multi_gpu:
            if self.args.defended_model == 'dgcnn' or self.args.defended_model == 'dgcnn_lh_mask_token' or self.args.defended_model == 'dgcnn_mexit_lh':
                projection_fp = self.classifier.module.linear3(feature_peak) # [B, ch]
            elif self.args.defended_model == 'curvenet' or self.args.defended_model == 'curvenet_lh_mask':
                projection_fp = self.classifier.module.conv2(feature_peak) # [B, ch]
            else:
                projection_fp = self.classifier.module.fc3(feature_peak) # [B, ch]

        else:
            if self.args.defended_model == 'dgcnn' or self.args.defended_model == 'dgcnn_lh_mask_token' or self.args.defended_model == 'dgcnn_mexit_lh':
                projection_fp = self.classifier.linear3(feature_peak) # [B, ch]
            elif self.args.defended_model == 'curvenet' or self.args.defended_model == 'curvenet_lh_mask':
                projection_fp = self.classifier.conv2(feature_peak) # [B, ch]
            else:
                projection_fp = self.classifier.fc3(feature_peak) # [B, ch]
           
        assert projection_fp is not None
        
        ####Frequency-aware group-level contrastive learning####
        loss_fregcl_h = self.GCL(zis=ori_net[-3], zjs=adv_net[-3], labels=self.target)
        loss_fregcl_m = self.GCL(zis=ori_net[-2], zjs=adv_net[-2], labels=self.target)
        loss_fregcl_l = self.GCL(zis=ori_net[-1], zjs=adv_net[-1], labels=self.target)
        loss_fregcl = (1-self.args.tem_m) * (loss_fregcl_h + loss_fregcl_l) + self.args.tem_m * loss_fregcl_m 
        ####Frequency-aware group-level contrastive learning####
        
        ####Frequency-aware instance-level contrastive learning####
        loss_freicl1_h = self.ICL1(query=ori_net[-3], pos=adv_net[-3], neg=ori_net_hard[-3])
        loss_freicl1_m = self.ICL1(query=ori_net[-2], pos=adv_net[-2], neg=ori_net_hard[-2])
        loss_freicl1_l = self.ICL1(query=ori_net[-1], pos=adv_net[-1], neg=ori_net_hard[-1])      
        loss_freicl1 = (1-self.args.tem_m) * (loss_freicl1_h + loss_freicl1_l) + self.args.tem_m * loss_freicl1_m
        
        loss_icl2_ori = self.ICL2(rep=projection_ori, fp=projection_fp, target=self.target)
        loss_icl2_adv = self.ICL2(rep=projection_adv, fp=projection_fp, target=self.target)
        loss_icl2 = loss_icl2_ori + loss_icl2_adv
        ####Frequency-aware instance-level contrastive learning####

        # Compute the classifier loss and update the classifier and projection head
        loss_cls = loss_fregcl + self.args.alpha * loss_icl2 + self.args.gamma * loss_freicl1 # nodeepincent
        self.optimizer_c.zero_grad()
        loss_cls.backward()  
        self.optimizer_c.step()


    def run_finetune(self, ii):
        """
        Fine-tune the fc layer.
        """
        self.classifier.eval()
        if self.args.use_multi_gpu:
            if self.args.defended_model == 'dgcnn_lh_mask_token':
                self.classifier.module.linear3.train()
            elif self.args.defended_model == 'curvenet_lh_mask_token':
                self.classifier.module.conv2.train()
            else:
                self.classifier.module.fc3.train()
        else:
            if self.args.defended_model == 'dgcnn_lh_mask_token':
                self.classifier.linear3.train()
            elif self.args.defended_model == 'curvenet_lh_mask_token':
                self.classifier.conv2.train()
            else:
                self.classifier.fc3.train()

        if ii:
           # self.criterion_fc = torch.nn.CrossEntropyLoss()
            self.criterion_fc = torch.nn.BCEWithLogitsLoss()
            if self.args.use_multi_gpu:
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    self.optimizer_fc = torch.optim.Adam(
                        self.classifier.module.linear3.parameters(),
                        lr=0.1,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=self.args.decay_rate
                    )
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    self.optimizer_fc = torch.optim.Adam(
                        self.classifier.module.conv2.parameters(),
                        lr=0.1,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=self.args.decay_rate
                    )
                else:
                    self.optimizer_fc = torch.optim.Adam(
                        self.classifier.module.fc3.parameters(),
                        lr=0.1,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=self.args.decay_rate
                    )
            else:
                if self.args.defended_model == 'dgcnn_lh_mask_token':
                    self.optimizer_fc = torch.optim.Adam(
                        self.classifier.linear3.parameters(),
                        lr=0.1,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=self.args.decay_rate
                    )
                elif self.args.defended_model == 'curvenet_lh_mask_token':
                    self.optimizer_fc = torch.optim.Adam(
                        self.classifier.conv2.parameters(),
                        lr=0.1,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=self.args.decay_rate
                    )
                else:
                    self.optimizer_fc = torch.optim.Adam(
                        self.classifier.fc3.parameters(),
                        lr=0.1,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=self.args.decay_rate
                    )
            self.scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_fc,
                T_0=5,
                T_mult=2
            )

        with torch.no_grad():
            features, _, x_l1, x_l2 = self.classifier(self.pc_ori)
        if self.args.use_multi_gpu:
            if self.args.defended_model == 'dgcnn_lh_mask_token':
                logits = self.classifier.module.linear3(features.detach())
            elif self.args.defended_model == 'curvenet_lh_mask_token':
                logits = self.classifier.module.conv2(features.detach())
            else:
                logits = self.classifier.module.fc3(features.detach())
        else:
            if self.args.defended_model == 'dgcnn_lh_mask_token':
                logits = self.classifier.linear3(features.detach())
            elif self.args.defended_model == 'curvenet_lh_mask_token':
                logits = self.classifier.conv2(features.detach())
            else:
                logits = self.classifier.fc3(features.detach())
                
        if self.args.dataset=='ShapeNetPart':
            targets_expanded = F.one_hot(self.target, num_classes=16).float()
        else:
            targets_expanded = F.one_hot(self.target, num_classes=40).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets_expanded)
        
        self.optimizer_fc.zero_grad()
        loss.backward()
        self.optimizer_fc.step()

    def run_test(self):
        """
        Evaluate the performance and robustness of CAT trained classifier.
        """
        self.classifier.eval()
        batch_size = self.target.size(0)
        ori_input = self.pc_ori.clone().detach().transpose(1, 2).contiguous() # [B, N, C]
        ori_target = self.target # [B]

        ############################################
        ### (1) clean acc
        ############################################
        with torch.no_grad():
            _, logits, x1,x2 = self.classifier(self.pc_ori)
            pred = torch.argmax(logits, dim=-1)
            acc_num = (pred == ori_target).sum().item()
        self.acc_clean.update(val=acc_num, n=batch_size)


        ############################################
        ### (2) noisy and random dropping attacks
        ############################################

        # Random Noisy Attack
        noisy_attack = JitterAttack(self.classifier, sigma=0.08, clip=0.32)
        _, acc_num = noisy_attack(ori_input.detach(), ori_target)
        self.acc_noisy.update(val=acc_num, n=batch_size)

        # Random Dropping Attack
        drop_attack = DropAttack(self.classifier, drop_num=800)
        _, acc_num = drop_attack(ori_input.detach(), ori_target)
        self.acc_drop.update(val=acc_num, n=batch_size)


        ############################################
        ### (3) untargeted adversarial attacks
        ############################################

        # setup attack settings
        # budget, step_size, number of iteration
        # settings adopted from CVPR'20 paper GvG-P
        delta = 0.02
        budget = delta * np.sqrt(self.args.input_point_nums * 3)  # \delta * \sqrt(N * d)
        num_iter = 10
        step_size = budget / float(num_iter)

        # which adv_func to use?
        if self.args.adv_func == 'logits':
            adv_func = LogitsAdvLoss(kappa=self.args.kappa, mode='untargeted')
        else:
            adv_func = CrossEntropyAdvLoss()

        # which clip_func to use?
        clip_func = ClipPointsL2(budget=budget)

        # which dist_func to use?
        dist_func = L2Dist()

        # FGM
        fgm_attack = FGM(self.classifier, adv_func=adv_func, budget=budget, dist_metric='l2')
        _, success_num = fgm_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_fgm_untar.update(val=success_num, n=batch_size)

        # I-FGM
        ifgm_attack = IFGM(self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                 step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = ifgm_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_ifgm_untar.update(val=success_num, n=batch_size)

        # MI-FGM
        mifgm_attack = MIFGM(self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                   step_size=step_size, num_iter=num_iter, mu=self.args.mu, dist_metric='l2')
        _, success_num = mifgm_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_mifgm_untar.update(val=success_num, n=batch_size)

        # PGD
        pgd_attack = PGD(self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                               step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = pgd_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_pgd_untar.update(val=success_num, n=batch_size)

        ############################################
        ### (4) targeted adversarial attacks
        ############################################

        # setup attack settings
        # budget, step_size, number of iteration
        # settings adopted from CVPR'20 paper GvG-P
        delta = self.args.budget
        budget = delta * np.sqrt(self.args.input_point_nums * 3)  # \delta * \sqrt(N * d)
        num_iter = int(self.args.num_iter)
        step_size = budget / float(num_iter)

        # generate the target label
        adv_target = []
        all_label_idx = [i for i in range(self.num_class)]
        for i in range(batch_size):
            idx = all_label_idx.copy()
            idx.remove(int(ori_target[i].cpu().item()))
            adv_target.append(np.random.choice(idx))
        assert len(adv_target) == batch_size
        adv_target = torch.Tensor(adv_target).to(ori_target.device) # [B]

        # which adv_func to use?
        if self.args.adv_func == 'logits':
            adv_func = LogitsAdvLoss(kappa=self.args.kappa, mode='targeted')
        else:
            adv_func = CrossEntropyAdvLoss()

        # FGM
        fgm_attack = FGM(self.classifier, adv_func=adv_func, budget=budget, dist_metric='l2')
        _, success_num = fgm_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_fgm_tar.update(val=success_num, n=batch_size)

        # I-FGM
        ifgm_attack = IFGM(self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                 step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = ifgm_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_ifgm_tar.update(val=success_num, n=batch_size)

        # MI-FGM
        mifgm_attack = MIFGM(self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                   step_size=step_size, num_iter=num_iter, mu=self.args.mu, dist_metric='l2')
        _, success_num = mifgm_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_mifgm_tar.update(val=success_num, n=batch_size)

        # PGD
        pgd_attack = PGD(self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                               step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = pgd_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_pgd_tar.update(val=success_num, n=batch_size)


    def show_results(self, mode):
        """
        Output the robustness performance of the given classifier.
        Support the following implement mode:
        - init: create new counting variables.
        - print: show the performance results.
        """
        if mode == 'init':
            # without any attacks
            self.acc_clean = AverageMeter()

            # noisy and random dropping attacks
            self.acc_noisy = AverageMeter()
            self.acc_drop = AverageMeter()

            # untargeted adversarial attacks
            self.success_fgm_untar = AverageMeter()
            self.success_ifgm_untar = AverageMeter()
            self.success_mifgm_untar = AverageMeter()
            self.success_pgd_untar = AverageMeter()

            # targeted adversarial attacks
            self.success_fgm_tar = AverageMeter()
            self.success_ifgm_tar = AverageMeter()
            self.success_mifgm_tar = AverageMeter()
            self.success_pgd_tar = AverageMeter()

        elif mode == 'print':
            # without any attacks
            self.log_string.write('\nWithout any attacks:')

            self.log_string.write('Clean ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_clean.sum, self.acc_clean.count, self.acc_clean.avg))

            # noisy and random dropping attacks
            self.log_string.write('\nInput preprocessing attacks:')

            self.log_string.write('Random noisy ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_noisy.sum, self.acc_noisy.count, self.acc_noisy.avg))
            self.log_string.write('Random dropping ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_drop.sum, self.acc_drop.count, self.acc_drop.avg))

            # untargeted adversarial attacks
            self.log_string.write('\nUntargeted adversarial attacks:')

            self.log_string.write('FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_fgm_untar.sum, self.success_fgm_untar.count, self.success_fgm_untar.avg))
            self.log_string.write('I-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_ifgm_untar.sum, self.success_ifgm_untar.count, self.success_ifgm_untar.avg))
            self.log_string.write('MI-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_mifgm_untar.sum, self.success_mifgm_untar.count, self.success_mifgm_untar.avg))
            self.log_string.write('PGD attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_pgd_untar.sum, self.success_pgd_untar.count, self.success_pgd_untar.avg))

            # targeted adversarial attacks
            self.log_string.write('\nTargeted adversarial attacks:')

            self.log_string.write('FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_fgm_tar.sum, self.success_fgm_tar.count, self.success_fgm_tar.avg))
            self.log_string.write('I-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_ifgm_tar.sum, self.success_ifgm_tar.count, self.success_ifgm_tar.avg))
            self.log_string.write('MI-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_mifgm_tar.sum, self.success_mifgm_tar.count, self.success_mifgm_tar.avg))
            self.log_string.write('PGD attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_pgd_tar.sum, self.success_pgd_tar.count, self.success_pgd_tar.avg))
        
        else:
            raise NotImplementedError


    def save_checkpoints(self, path, epoch_c, epoch_ng, mode):
        """
        Save model checkpoints.
        """
        if mode == 'best':
            if self.args.use_multi_gpu:
                save_checkpoint(epoch_c, self.classifier, path, modelnet='best-cls', use_multi_gpu=True)
                save_checkpoint(epoch_ng, self.noise_generator, path, modelnet='best-ng', use_multi_gpu=True)
            else:
                save_checkpoint(epoch_c, self.classifier, path, modelnet='best-cls', use_multi_gpu=False)
                save_checkpoint(epoch_ng, self.noise_generator, path, modelnet='best-ng', use_multi_gpu=False)
        else:
            if self.args.use_multi_gpu:
                save_checkpoint(epoch_c, self.classifier, path, modelnet='latest-cls', use_multi_gpu=True)
                save_checkpoint(epoch_ng, self.noise_generator, path, modelnet='latest-ng', use_multi_gpu=True)
            else:
                save_checkpoint(epoch_c, self.classifier, path, modelnet='latest-cls', use_multi_gpu=False)
                save_checkpoint(epoch_ng, self.noise_generator, path, modelnet='latest-ng', use_multi_gpu=False)