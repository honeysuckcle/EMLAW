import torch.nn as nn
import torch
import torch.nn.functional as F

def LogitNorm(x, dim=-1, temp = 0.01):
    norms = torch.norm(x, p=2, dim=dim, keepdim=True) + 1e-7
    logit_norm = torch.div(x, norms) / temp
    return logit_norm