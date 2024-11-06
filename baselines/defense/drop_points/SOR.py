"""SOR defense proposed by ICCV'19 paper DUP-Net"""
import numpy as np
import torch
import torch.nn as nn


class SORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, k=2, alpha=1.1, npoint=1024):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha
        self.npoint = npoint

    def outlier_removal(self, x):
        """Removes large kNN distance points.

        Args:
            x (torch.FloatTensor): batch input pc, [B, K, 3]

        Returns:
            torch.FloatTensor: pc after outlier removal, [B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        threshold = mean + self.alpha * std  # [B]
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = [x[i][bool_mask[i]] for i in range(B)]
        return sel_pc

    def process_data(self, pc, npoint=None):
        """Process point cloud data to be suitable for
            PU-Net input.
        We do two things:
            sample npoint or duplicate to npoint.

        Args:
            pc (torch.FloatTensor): list input, [(N_i, 3)] from SOR.
                Need to pad or trim to [B, self.npoint, 3].
        """
        if npoint is None:
            npoint = self.npoint
        B = len(pc)
        proc_pc = torch.zeros((B, npoint, 3)).float().cuda()
        for pc_idx in range(B):
            one_pc = pc[pc_idx]
            # [N_i, 3]
            N = len(one_pc)
            if N > npoint:
                # random sample some of them
                idx = np.random.choice(N, npoint, replace=False)
                idx = torch.from_numpy(idx).long().cuda()
                one_pc = one_pc[idx]
            elif N < npoint:
                # just duplicate to the number
                duplicated_pc = one_pc
                num = npoint // N - 1
                for i in range(num):
                    duplicated_pc = torch.cat([
                        duplicated_pc, one_pc
                    ], dim=0)
                num = npoint - len(duplicated_pc)
                # random sample the remaining
                idx = np.random.choice(N, num, replace=False)
                idx = torch.from_numpy(idx).long().cuda()
                one_pc = torch.cat([
                    duplicated_pc, one_pc[idx]
                ], dim=0)
            proc_pc[pc_idx] = one_pc
        return proc_pc

    def forward(self, x):
        with torch.no_grad():
            x = x.transpose(1, 2)
            x = self.outlier_removal(x)
            x = self.process_data(x)  # to batch input
            x = x.transpose(1, 2)
        return x
