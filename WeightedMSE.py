import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


filt = -1 * np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

filt = torch.Tensor(filt).cuda()
filt = filt.unsqueeze(0).unsqueeze(0).cuda() / 4


def cal_dist(mask, max_pool):
    level = len(max_pool) + 1
    mask = mask.float()
    edge = (F.conv3d(mask, filt, padding=1) > 0).float()
    dist = torch.zeros_like(edge).float().cuda()
    for mpk in max_pool:
        dist += mpk(edge)
    dist = (level - dist) / level
    dist[mask == 1] = 0
    return dist


def generate_pooling(level=10):
    max_pool = []
    for k in range(0, level+1):
        d = 2 * k + 1
        max_pool.append(torch.nn.MaxPool3d([1, d, d], 1, [0, k, k]).cuda())
    return max_pool


def cal_true_label_weight(tags):
    weight = torch.ones_like(tags).float()
    weight[tags == 0] = torch.mean(tags.float()) + 0.01
    weight[tags == 1] = torch.mean((tags == 0).float()) + 0.01
    return weight


def cal_pred_label_weight(preds):
    preds_int = preds.round().detach()
    return cal_true_label_weight(preds_int)


class WeightedMSE(nn.Module):
    def __init__(self, k=2, level=20):
        super(WeightedMSE, self).__init__()
        self.k = k
        self.max_pool = generate_pooling(level)

    def forward(self, outputs, targets):
        batchsize = outputs.size(0)

        diff = torch.clamp(outputs - targets, min=0)
        dist = cal_dist(targets, self.max_pool)
        weight = torch.pow(dist, self.k)

        diff = diff * weight
        diff = torch.pow(diff, 2)

        batch_loss = diff.view(batchsize, -1)
        weight = weight.view(batchsize, -1)
        weight_s = weight.sum(1).unsqueeze(1)
        batch_loss = batch_loss / weight_s
        batch_loss = batch_loss.sum(1).mean()

        return batch_loss