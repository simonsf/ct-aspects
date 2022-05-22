import torch
import torch.nn as nn



class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def forward(self, output1, output2, tag1, tag2, weight1, weight2):
        #weight = torch.zeros_like(tag1)
       # weight[tag1 == 1] = weight1[tag1 == 1]
        #weight[tag2 == 1] = weight2
        weight = (tag1 * weight1 + tag2 * weight2).float()
        weight = weight.detach()
        #print(weight.squeeze(1))
        diff1 = torch.pow(torch.clamp(output2 - output1 + 0.5, min=0), 2) * tag1
        diff2 = torch.pow(torch.clamp(output1 - output2 + 0.5, min=0), 2) * tag2
        loss = torch.sum((diff1 + diff2) * weight)/(torch.sum(weight) + 0.001)

        weight_inv = (tag1 == 0) * (tag2 == 0)
        weight_inv = weight_inv.float()
        neg_left = torch.pow(torch.clamp(output1 - 0.2, min=0), 2) * weight_inv * weight1
        neg_right = torch.pow(torch.clamp(output2 - 0.2, min=0), 2) * weight_inv * weight2
        neg_loss = torch.sum(neg_left) / torch.sum(weight_inv * weight1) \
                   + torch.sum(neg_right) / torch.sum(weight_inv * weight2)
        return loss + neg_loss / 1.5