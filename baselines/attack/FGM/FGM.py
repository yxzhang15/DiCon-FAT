"""Implementation of gradient based attack methods, FGM, I-FGM, MI-FGM, PGD, etc.
Related paper: CVPR'20 GvG-P,
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Self-Robust_3D_Point_Recognition_via_Gather-Vector_Guidance_CVPR_2020_paper.pdf
"""

import torch
import numpy as np


class FGM:
    """Class for FGM attack.
    """

    def __init__(self, source_model, target_model=None, adv_func=None, budget=0.08, dist_metric='l2'):
        """FGM attack.

        Args:
            source_model (torch.nn.Module): source model
            target_model (torch.nn.Module): target model
            adv_func (function): adversarial loss function
            budget (float): \epsilon ball for FGM attack
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """

        self.model = source_model.cuda()
        if target_model is None:
            self.target_model = source_model.cuda()
        else:
            self.target_model = target_model.cuda()
        self.model.eval()
        self.target_model.eval()

        self.adv_func = adv_func
        self.budget = budget
        self.dist_metric = dist_metric.lower()

    def get_norm(self, x):
        """Calculate the norm of a given data x.

        Args:
            x (torch.FloatTensor): [B, 3, K]
        """
        if self.dist_metric == 'l1':
            # use global l1 norm here!
            norm = torch.sum(torch.abs(x), dim=[1, 2])
        else:
            # use global l2 norm here!
            norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm

    def get_gradient(self, data, target, normalize=True):
        """Generate one step gradient.

        Args:
            data (torch.FloatTensor): batch pc, [B, 3, K]
            target (torch.LongTensor): target label, [B]
            normalize (bool, optional): whether l2 normalize grad. Defaults to True.
        """
        data = data.float().cuda()
        data.requires_grad_()
        target = target.long().cuda()

        # forward pass
        logits = self.model(data)
        if isinstance(logits, tuple):
            logits = logits[1]  # [B, class]
        pred = torch.argmax(logits, dim=-1)  # [B]

        # backward pass
        loss = self.adv_func(logits, target).mean()
        loss.backward()
        with torch.no_grad():
            grad = data.grad.detach()  # [B, 3, K]
            if normalize:
                if self.dist_metric == 'linf':
                    grad = torch.sign(grad)
                else:
                    norm = self.get_norm(grad)
                    grad = grad / (norm[:, None, None] + 1e-9)
        return grad, pred

    def attack(self, data, target, mode='targeted'):
        """One step FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label, [B]
        """
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        target = target.long().cuda()

        # gradient
        normalized_grad, _ = self.get_gradient(pc, target)  # [B, 3, K]
        perturbation = normalized_grad * self.budget
        with torch.no_grad():
            perturbation = perturbation.transpose(1, 2).contiguous()
            data = data - perturbation  # no need to clip
            
            # test attack performance
            logits = self.target_model(data.transpose(1, 2).contiguous())
            if isinstance(logits, tuple):
                logits = logits[1]
            pred = torch.argmax(logits, dim=-1)
            if mode.lower() == 'targeted':
                success_num = (pred == target).sum().item()
            else:
                success_num = (pred != target).sum().item()

        # print('Final success {}/{}'.format(success_num, data.shape[0]))
        torch.cuda.empty_cache()
       # return data.detach().cpu().numpy(), success_num
        return data, success_num


class IFGM(FGM):
    """Class for I-FGM attack.
    """

    def __init__(self, source_model, target_model=None, adv_func=None, clip_func=None,
                 budget=0.08, step_size=0.008, num_iter=20,
                 dist_metric='l2'):
        """Iterative FGM attack.

        Args:
            source_model (torch.nn.Module): source model
            target_model (torch.nn.Module): target model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(IFGM, self).__init__(source_model, target_model, adv_func, budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter

    def attack(self, data, target, mode='targeted'):
        """Iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            normalized_grad, pred = self.get_gradient(pc, target)

            # success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                # print('iter {}/{}, success: {}/{}'.format(
                #     iteration, self.num_iter, success_num, B))
                torch.cuda.empty_cache() #change
            perturbation = self.step_size * normalized_grad

            # add perturbation and clip
            with torch.no_grad():
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        # end of iteration
        with torch.no_grad():
            logits = self.target_model(pc)
            if isinstance(logits, tuple):
                logits = logits[1]
            pred = torch.argmax(logits, dim=-1)
            if mode.lower() == 'targeted':
                success_num = (pred == target).sum().item()
            else:
                success_num = (pred != target).sum().item()
        # print('Final success: {}/{}'.format(success_num, B))
      #  return pc.transpose(1, 2).contiguous().detach().cpu().numpy(), success_num
        return pc.transpose(1, 2).contiguous().detach(), success_num


class MIFGM(FGM):
    """Class for MI-FGM attack.
    """

    def __init__(self, source_model, target_model=None, adv_func=None, clip_func=None,
                 budget=0.08, step_size=0.008, num_iter=20, mu=1.,
                 dist_metric='l2'):
        """Momentum enhanced iterative FGM attack.

        Args:
            source_model (torch.nn.Module): source model
            target_model (torch.nn.Module): target model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            mu (float): momentum factor
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(MIFGM, self).__init__(source_model, target_model, adv_func,
                                    budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.mu = mu

    def attack(self, data, target, mode='targeted'):
        """Momentum enhanced iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()
        momentum_g = torch.tensor(0.).cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            grad, pred = self.get_gradient(pc, target, normalize=False)
            # success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                # print('iter {}/{}, success: {}/{}'.format(
                #     iteration, self.num_iter, success_num, B))
                torch.cuda.empty_cache()

            # grad is [B, 3, K]
            # normalized by l1 norm
            grad_l1_norm = torch.sum(torch.abs(grad), dim=[1, 2])  # [B]
            normalized_grad = grad / (grad_l1_norm[:, None, None] + 1e-9)
            momentum_g = self.mu * momentum_g + normalized_grad
            if self.dist_metric == 'linf':
                normalized_g = torch.sign(momentum_g)
            else:
                g_norm = self.get_norm(momentum_g)
                normalized_g = momentum_g / (g_norm[:, None, None] + 1e-9)
            perturbation = self.step_size * normalized_g

            # add perturbation and clip
            with torch.no_grad():
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        # end of iteration
        with torch.no_grad():
            logits = self.target_model(pc)
            if isinstance(logits, tuple):
                logits = logits[1]
            pred = torch.argmax(logits, dim=-1)
            if mode.lower() == 'targeted':
                success_num = (pred == target).sum().item()
            else:
                success_num = (pred != target).sum().item()
        # print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach().cpu().numpy(), success_num


class PGD(IFGM):
    """Class for PGD attack.
    """

    def __init__(self, source_model, target_model=None, adv_func=None, clip_func=None,
                 budget=0.08, step_size=0.008, num_iter=20,
                 dist_metric='l2'):
        """PGD attack.

        Args:
            source_model (torch.nn.Module): source model
            target_model (torch.nn.Module): target model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(PGD, self).__init__(source_model, target_model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric)

    def attack(self, data, target, mode='targeted'):
        """PGD attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label, [B]
        """
        # the only difference between IFGM and PGD is
        # the initialization of noise
        epsilon = self.budget / \
            ((data.shape[1] * data.shape[2]) ** 0.5)
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation
        return super(PGD, self).attack(init_data, target, mode)

class PGD_AT(IFGM):
    """Class for PGD attack.
    """

    def __init__(self, source_model, target_model=None, adv_func=None, clip_func=None,
                 budget=0.08, step_size=0.008, num_iter=20,
                 dist_metric='l2'):
        """PGD attack.

        Args:
            source_model (torch.nn.Module): source model
            target_model (torch.nn.Module): target model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(PGD_AT, self).__init__(source_model, target_model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric)

    def attack(self, data, target, mode='targeted'):
        """PGD attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label, [B]
        """
        # the only difference between IFGM and PGD is
        # the initialization of noise
        epsilon = self.budget / \
            ((data.shape[1] * data.shape[2]) ** 0.5)
        epsilon = 0.04
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation
        return super(PGD_AT, self).attack(init_data, target, mode)