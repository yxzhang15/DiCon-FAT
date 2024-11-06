"""Random Dropping Attack"""
import numpy as np

import torch
import torch.nn as nn


class DropAttack(nn.Module):
    """Random dropping points as attack.
    """

    def __init__(self, model, drop_num=500):
        """Dropping attack method.

        Args:
            drop_num (int, optional): number of points to drop.
                                        Defaults to 500.
        """
        super(DropAttack, self).__init__()

        self.model = model.cuda()
        self.model.eval()

        self.drop_num = drop_num


    def random_drop(self, pc):
        """Random drop self.drop_num points in each pc.

        Args:
            pc (torch.FloatTensor): batch input pc, [B, K, 3]
        """
        B, K = pc.shape[:2]
        idx = [np.random.choice(K, K - self.drop_num, replace=False) for _ in range(B)]
        pc = torch.stack([pc[i][torch.from_numpy(idx[i]).long().to(pc.device)] for i in range(B)])
        return pc


    def forward(self, x, target):
        with torch.no_grad():
            x = self.random_drop(x)
            logits = self.model(x.transpose(1, 2).contiguous())
            if isinstance(logits, tuple):
                logits = logits[1]
            pred = torch.argmax(logits, dim=-1)
            acc_num = (pred == target).sum().item()
        torch.cuda.empty_cache()
        return x.detach().cpu().numpy(), acc_num
