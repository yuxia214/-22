import torch.nn as nn
import torch.nn.functional as F

# classification loss
class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1) # [n_samples, n_classes]
        target = target.long()        # [n_samples]
        loss = self.loss(pred, target) / len(pred)
        return loss

# regression loss
class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        loss = F.smooth_l1_loss(pred, target, reduction='sum', beta=self.beta) / len(pred)
        return loss
