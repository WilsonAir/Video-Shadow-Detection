import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable



def scotch_loss(out_1, out_2, temperature=0.1, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """
    out_1_dist = out_1
    out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # positive similarity: using out_1 with itself, so only consider the first batch_size elements
    pos = torch.exp(torch.sum(out_1 * out_1, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + pos + eps)).mean()

    return loss