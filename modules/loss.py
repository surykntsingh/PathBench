import torch
import torch.nn as nn

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def get_attn_regularization(weights, eps=1e-8):
    # Attention Regularization

    entropy = - (weights * (weights + eps).log()).sum(dim=-1)  # [B, L, H]
    return entropy.mean()


def compute_hybrid_loss(output, reports_ids, reports_masks, weights, g_lambda):
    nll_loss = compute_nll_loss(output, reports_ids, reports_masks)
    reg_g = get_attn_regularization(weights)
    return nll_loss + g_lambda * reg_g

def compute_nll_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss