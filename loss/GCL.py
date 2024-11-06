import torch
import numpy as np


class GCL(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(GCL, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, labels):
        labels = labels.contiguous().view(-1, 1) # [N, 1]
        mask = torch.eq(labels, labels.T).float() # [N, N]
        mask = mask.repeat(2, 2) # [2N, 2N]
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(self.batch_size * 2).view(-1, 1).to(self.device),
            0
        ) # [2N, 2N]
        mask = mask * logits_mask # [2N, 2N]
        return mask, logits_mask

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

    def forward(self, zis, zjs, labels):
        representations = torch.cat([zjs, zis], dim=0) # [2N, C]
        mask, logits_mask = self._get_correlated_mask(labels)

        similarity_matrix = self.similarity_function(representations, representations) # [2N, 2N]
        logits = torch.div(similarity_matrix, self.temperature) # [2N, 2N]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # [2N, 2N]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # [2N, 2N]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # [2N]

        # compute loss
        loss = -1 * mean_log_prob_pos.mean() # [1]

        return loss
