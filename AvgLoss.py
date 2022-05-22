import torch
import torch.nn as nn


class AvgLoss(nn.Module):
    def __init__(self):
        super(AvgLoss, self).__init__()

    def forward(self, out1, out2):
        dot_mult = out1 * out2
        avgloss = dot_mult.mean()
        return avgloss
