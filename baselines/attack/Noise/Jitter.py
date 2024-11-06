"""Jittering Attack"""
import numpy as np

import torch
import torch.nn as nn


class JitterAttack(nn.Module):
    """Random jittering points as attack.
    """

    def __init__(self, model, sigma=0.01, clip=0.05):
        """Jittering Attack method.

        Args:
            sigma (float, optional): variance of noises to add.
                                        Defaults to 0.01.
            clip (float, optional): bound value of noises to be constrained.
                                        Defaults to 0.05.
        """
        super(JitterAttack, self).__init__()

        self.model = model.cuda()
        self.model.eval()

        self.sigma = sigma
        self.clip = clip


    def jitter_point_cloud(self, batch_data):
        """Random jitter each point in each pc.

        Args:
            pc (torch.FloatTensor): batch input pc, [B, K, 3]
        """
        B, N, C = batch_data.shape
        assert (self.clip > 0)
        jittered_data = torch.clamp(self.sigma * torch.randn(B, N, C), min=-1 * self.clip, max=self.clip).cuda()
        jittered_data += batch_data
        return jittered_data


    def forward(self, x, target):
        with torch.no_grad():
            x = self.jitter_point_cloud(x)
            logits = self.model(x.transpose(1, 2).contiguous())
            if isinstance(logits, tuple):
                logits = logits[1]
            pred = torch.argmax(logits, dim=-1)
            acc_num = (pred == target).sum().item()
        torch.cuda.empty_cache()
        return x.detach().cpu().numpy(), acc_num
