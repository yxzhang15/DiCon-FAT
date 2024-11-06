# -*- coding: utf-8 -*-

import argparse

class Arguments:
    def __init__(self, stage='train'):
        self._parser = argparse.ArgumentParser(description='DiCon-FAT.')

        self.add_common_args()
        if stage == 'train':
            self.add_train_args()
        else:
            self.add_test_args()

    def add_common_args(self):
        ### path related
        self._parser.add_argument('--experiment_dir', type=str, default='DiCon-FAT')

        ### data related
        self._parser.add_argument('--batch_size', type=int, default=64, help='Batch size in training')
        self._parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size in training')
        self._parser.add_argument('--dataset', type=str, default='ModelNet40', choices=['ModelNet40', 'ShapeNetPart'])
        self._parser.add_argument('--data_path', type=str, default='./data/modelnet40_normal_resampled/')
        self._parser.add_argument('--input_point_nums', type=int, default=1024, help='Point nums of each point cloud')
        self._parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
        self._parser.add_argument('--num_workers', type=int, default=4, help='Worker nums of data loading.')

        ### network related
        self._parser.add_argument('--defended_model', type=str, default='pointnet_lh_mask_token', choices=['pointnet2_lh_mask_token', 'pointnet_lh_mask_token', 'dgcnn_lh_mask_token', 'curvenet_lh_mask_token'])
        self._parser.add_argument('--decoder_type', type=str, default='normal_conv', choices=['normal_conv', 'treegcn'])
        
        ### TreeGAN architecture related
        self._parser.add_argument('--DEGREE', type=int, default=[1, 2, 2, 2, 2, 2, 32], nargs='+', help='Upsample degrees for generator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3, 64, 128, 256, 256, 512], nargs='+', help='Features for discriminator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
        self._parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        
        ### others
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        self._parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='Whether to use more than 1 gpu [default: False]')


    def add_train_args(self):
        ### training related
        self._parser.add_argument('--epochs', type=int, default=160, help='nums of training epochs.')
        self._parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
        self._parser.add_argument('--lr_c', default=0.001, type=float, help='learning rate for classifier in training [default: 0.001]')
        self._parser.add_argument('--lr_ng', default=0.001, type=float, help='learning rate for noise-generator in training [default: 0.001]')
        self._parser.add_argument('--lr_fp', default=0.001, type=float, help='learning rate for prototype searching [default: 0.001]')
        self._parser.add_argument('--epoch_update_fp', type=int, default=1, help='nums of epochs to update fp.')
        self._parser.add_argument('--inner_loop_nums', type=int, default=4, help='nums of inner loop.')
        self._parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
        self._parser.add_argument('--eps', type=float, default=0.04, help='the intensity epsilon of added perturbation')
        self._parser.add_argument('--tm_gcl', type=float, default=0.1, help='the temperature of gcl loss')
        self._parser.add_argument('--tm_icl1', type=float, default=0.1, help='the temperature of icl1 loss')
        self._parser.add_argument('--tm_icl2', type=float, default=0.25, help='the temperature of icl2 loss')
        self._parser.add_argument('--tm_adv', type=float, default=0.1, help='the temperature of adversarial loss')

        self._parser.add_argument('--use_cosine_similarity', action='store_true', default=True)
        self._parser.add_argument('--alpha', type=float, default=1., help='the hyper-parameter in the frecl loss')
        self._parser.add_argument('--beta', type=float, default=1., help='the hyper-parameter in the generate adv loss')
        self._parser.add_argument('--gamma', type=float, default=0.01, help='the hyper-parameter in the frecl loss')
        self._parser.add_argument('--tem_m', type=float, default=0.1, help='the hyper-parameter in the frecl los')
        self._parser.add_argument('--init_search_iters', type=int, default=500, help='initial num of iterations for prototype searching.')
        self._parser.add_argument('--update_search_iters', type=int, default=10, help='per-step num of iterations for prototype searching.')


        ### testing related
        self._parser.add_argument('--adv_func', type=str, default='logits', choices=['logits', 'cross_entropy'])
        self._parser.add_argument('--kappa', type=float, default=0., help='min margin in logits adv loss')
        self._parser.add_argument('--budget', type=float, default=0.08, help='FGM attack budget')
        self._parser.add_argument('--num_iter', type=int, default=50, help='I-FGM iterate step')
        self._parser.add_argument('--mu', type=float, default=1., help='momentum factor for MI-FGM attack')
        self._parser.add_argument('--attack_lr', type=float, default=1e-2, help='lr in CW optimization')
        self._parser.add_argument('--binary_step', type=int, default=10, metavar='N', help='Binary search step')
        self._parser.add_argument('--num_iter_cw', type=int, default=50, metavar='N', help='Number of iterations in each CW search step')


    def add_test_args(self):
        self._parser.add_argument('--mode', type=str, default='test_normal', choices=['test_normal', 'test_aa', 'test_ba', 'test_ng'])
        self._parser.add_argument('--checkpoint_dir', type=str, default='')
        self._parser.add_argument('--ng_dir', type=str, default='')

        ### testing related
        self._parser.add_argument('--adv_func', type=str, default='logits', choices=['logits', 'cross_entropy'])
        self._parser.add_argument('--kappa', type=float, default=0., help='min margin in logits adv loss')
        self._parser.add_argument('--budget', type=float, default=0.08, help='FGM attack budget')
        self._parser.add_argument('--num_iter', type=int, default=50, help='I-FGM iterate step')
        self._parser.add_argument('--mu', type=float, default=1., help='momentum factor for MI-FGM attack')
        self._parser.add_argument('--attack_lr', type=float, default=1e-2, help='lr in CW optimization')
        self._parser.add_argument('--binary_step', type=int, default=10, metavar='N', help='Binary search step')
        self._parser.add_argument('--num_iter_cw', type=int, default=50, metavar='N', help='Number of iterations in each CW search step')

        # for pre-defense settings
        self._parser.add_argument('--use_pre_defense', action='store_true', default=False)
        self._parser.add_argument('--pre_defense', type=str, default='sor', choices=['sor', 'srs', 'dupnet', 'upsampling', 'denoising'])
        # for black-box attacks
        self._parser.add_argument('--source_model', type=str, default='pointnet_cls', choices=['pointnet_cls', 'pointnet_lh_mask_token', 'pointnet2_cls_msg', 'pointnet2_lh_mask','dgcnn', 'dgcnn_lh_mask_token','curvenet','curvenet_lh_mask'])
        self._parser.add_argument('--source_model_wo_defense', action='store_true', default=False)
        self._parser.add_argument('--source_model_dir', type=str, default='')
        

            



    def parser(self):
        return self._parser
    


    
   


