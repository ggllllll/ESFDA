
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, weight=None):
        smooth = 1        #防止分母为0

        size = pred.size(0)         #获取batchsize的大小

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)          #将预测张量和目标张量展平成为二维张量

        intersection = pred_flat * target_flat       #计算它们之间的交集
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)       #计算dice值
        if weight is not None:
            #print(weight)
            dice_score = weight * dice_score         #将权重与dice相乘重新得到dice的值
            dice_loss = weight.sum() /size - dice_score.sum()/size       #计算dice_loss
        else:
            dice_loss = 1 - dice_score.sum()/size

        return dice_loss


def smooth_loss(y_true, y_pred, weight=None):
    H = y_pred.shape[-2]     #得到分割图像上的高
    W = y_pred.shape[-1]     #得到分割图像上的宽

    #计算损失
           #计算水平(后两行）垂直（前两行）方向上相邻像素点差值
    loss = torch.abs(y_pred[:, 1:H - 1, 1:W - 1] - y_pred[:, 0:H - 2, 1:W - 1]) + \
           torch.abs(y_pred[:, 1:H - 1, 1:W - 1] - y_pred[:, 2:H, 1:W - 1]) + \
           torch.abs(y_pred[:, 1:H - 1, 1:W - 1] - y_pred[:, 1:H - 1, 0:W - 2]) + \
           torch.abs(y_pred[:, 1:H - 1, 1:W - 1] - y_pred[:, 1:H - 1, 2:W])
          #计算真是标签中平滑函数的权重系数M1、M2、M3、M4
    M1 = (y_true[:, 1:H - 1, 1:W - 1] == y_true[:, 0:H - 2, 1:W - 1]).to(dtype=torch.float32)     #计算垂直方向上的相邻像素之间的权重
    M2 = (y_true[:, 1:H - 1, 1:W - 1] == y_true[:, 2:H, 1:W - 1]).to(dtype=torch.float32)         #计算垂直方向上的相邻像素之间的权重
    M3 = (y_true[:, 1:H - 1, 1:W - 1] == y_true[:, 1:H - 1, 0:W - 2]).to(dtype=torch.float32)     #计算水平方向上相邻像素之间的权重
    M4 = (y_true[:, 1:H - 1, 1:W - 1] == y_true[:, 1:H - 1, 2:W]).to(dtype=torch.float32)         #计算水平方向上相邻像素之间的权重

    mask = M1 * M2 * M3 * M4          #将四个权重系数相乘得到一个新的掩码  #
    loss = loss * mask                #预测计算的损失与掩码相乘
    loss = loss.mean(dim=[1,2])       #计算的损失在第一第二个维度上取平均值
    if weight is not None:
        loss = loss * weight          #将损失在与权重相乘

    loss = loss.mean()                #计算损失的平均值

    return loss

"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


""" Entropy Minimization"""
class softCrossEntropy(nn.Module):
    def __init__(self, ignore_index= -1):
        super(softCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss
        """
        assert inputs.size() == target.size()
        mask = (target != self.ignore_index)

        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.mean(torch.mul(-log_likelihood, target)[mask])

        return loss


"""Maxsquare Loss"""
class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1):
        super().__init__()
        self.ignore_index = ignore_index
        #self.num_class = num_class
    
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        #mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2) + torch.pow(1-prob, 2)) / 2
        return loss
