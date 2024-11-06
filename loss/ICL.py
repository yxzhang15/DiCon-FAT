import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class ICL1(torch.nn.Module):
    def __init__(self, temperature, use_cosine_similarity=True):
        super(ICL1, self).__init__()
        self.temperature = temperature
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity
        else:
            return self._dot_similarity

    @staticmethod
    def _dot_simililarity(x, y):
        # x shape: (2N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (2N, 2N)
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (2N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (2N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    def forward(self, query, pos, neg):
        self.batch_size = query.size(0)
        criterion_global = nn.CrossEntropyLoss()
        view_qp = torch.matmul(query, pos.transpose(0,1)) # [B, B]
        view_qn = torch.matmul(query, neg.transpose(0,1)) # [B, B]

        view_score = torch.cat(tensors=(view_qp, view_qn), dim=1)   # [B, B+H]
        view_score = view_score / self.temperature

        target_view_score = torch.arange(view_score.shape[0]).to('cuda') 
        loss_global = criterion_global(view_score, target_view_score)

        return loss_global / (self.batch_size)

class ICL2(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(ICL2, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, num_class, batch_index):
        # num_class shape: (1)
        # batch_index shape: (N)
        diag = np.eye(num_class) # [M, M]
        mask_pos = torch.from_numpy((diag)).to(self.device)
        mask_pos = mask_pos.index_select(0, batch_index) # [N, M]
        mask_neg = (1 - mask_pos).type(torch.bool)
        mask_pos = mask_pos.type(torch.bool)
        return mask_pos, mask_neg

    @staticmethod
    def _dot_simililarity(x, y):
        # x shape: (N, 1, C)
        # y shape: (1, C, N)
        # v shape: (N, N)
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, rep, fp, target):
        # rep shape: (N, C)
        # fp shape: (M, C)
        # target shape: (N)

        # compute similarity matrix
        similarity_rep2fp = self.similarity_function(rep, fp) # [N, M]
        similarity_fp2rep = self.similarity_function(fp, rep).T # [N, M]

        # get masks from the batch labels
        mask_pos, mask_neg = self._get_correlated_mask(fp.shape[0], target)

        # filter out the scores from the positive samples
        l_pos = similarity_rep2fp[mask_pos] # [N]
        r_pos = similarity_fp2rep[mask_pos] # [N]
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) # [2N, 1]

        # filter out the scores from the negative samples
        l_neg = similarity_rep2fp[mask_neg].view(self.batch_size, -1) # [N, M-1]
        r_neg = similarity_fp2rep[mask_neg].view(self.batch_size, -1) # [N, M-1]
        negatives = torch.cat([l_neg, r_neg]) # [2N, M-1]

        logits = torch.cat((positives, negatives), dim=1) # [2N, M]
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long() # [2N]
        loss = self.criterion(logits, labels) # [1]

        return loss / (2 * self.batch_size)