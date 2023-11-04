import torch
from torch import nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        print("BCE Dice Loss")

    def forward(self, inputs, targets, smooth=1e-3):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    

class semiSupervisedBCELoss(nn.Module):
    def __init__(self, lambda_coef=0.01,net_list=[]):
        super(semiSupervisedBCELoss, self).__init__()
        assert len(net_list)!=0
        self.nets=net_list
        for net in self.nets:
            net.eval()
            net.requires_grad_(False)
        self.lambda_coeff=lambda_coef
        
    def forward(self,x,yp,y,m):
        mask=(m[:,0]!=0)
        inputs=(F.sigmoid(yp)[(m[:,0]!=0)]).to(torch.float)
        
        ym=y.sum(axis=1)!=0
        y=y[:,2]==0
        y=y[ym]
        yp=yp[ym]
        
        mask=mask[ym]
        yp=yp[mask]
        y=y[mask]
        
        supervised_loss=F.binary_cross_entropy(F.sigmoid(yp), y.to(torch.float), reduction='mean')
        y_nets=None
        
        for net in self.nets:
            y_net=net(x)
            if y_nets is None:
                y_nets=y_net
            else :
                y_nets+=y_net
        y_nets=y_nets>=0
        
        unsupervised_loss=F.binary_cross_entropy(inputs, y_nets[(m[:,0]!=0)].to(torch.float), reduction='mean')
        
        loss=supervised_loss+self.lambda_coeff*unsupervised_loss
        
        return loss