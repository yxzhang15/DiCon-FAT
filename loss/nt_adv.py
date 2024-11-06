import torch
import numpy as np


class NTAdvLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, beta, use_cosine_similarity):
        super(NTAdvLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.beta = beta
        self.device = device

        self.similarity_function = self._get_similarity_function(use_cosine_similarity)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        # x shape: (N, C)
        # y shape: (C, N)
        # v shape: (N)
        v = torch.diag(torch.tensordot(x, y.T, dims=1))
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, C)
        # y shape: (N, C)
        # v shape: (N)
        v = self._cosine_similarity(x, y)
        return v

    def forward(self, ori, adv, fp):
        # ori shape: (N, C)
        # adv shape: (N, C)
        # fp shape: (N, C)
        similarity_ori2adv = self.similarity_function(ori, adv) # [N]
        similarity_adv2ori = self.similarity_function(adv, ori) # [N]
        loss1 = torch.cat([similarity_ori2adv, similarity_adv2ori]) # [2N]
        loss1 /= self.temperature # [2N]

        similarity_fp2adv = self.similarity_function(fp, adv) # [N]
        similarity_adv2fp = self.similarity_function(adv, fp) # [N]
        loss2 = torch.cat([similarity_fp2adv, similarity_adv2fp]) # [2N]
        loss2 /= self.temperature # [2N]

        loss = - torch.exp(-loss1) - self.beta * torch.exp(-loss2) # [2N]
        # loss = torch.exp(-loss1) - self.beta * torch.exp(-loss2) # [2N]
        # loss = loss1 + self.beta * loss2 # [2N]
        loss = torch.sum(loss) # [1]

        return loss / (2 * self.batch_size)