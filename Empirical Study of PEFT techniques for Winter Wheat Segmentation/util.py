from torch import nn
import torch
import numpy as np
from sklearn import metrics
import random
import os
from segmentation_models_pytorch.metrics import iou_score,f1_score,get_stats

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    (copied from HuggingFace)
    returns the number of trainable parameters, total number of paramters, and the percentage of trainable parametrs
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return(trainable_params,all_param,100 * trainable_params / all_param)

def get_trainable_parameters(model):
    """
    return all trainable parameters in the model.
    """
    trainable_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
          trainable_params.append(param)
    return trainable_params

# class HeatMap:
#     """return a heat map (currently not used)"""
#     d={}
#     def __call__(self, y,yp) :
#       y=int(y)
#       yp=int(yp)
#       if (y,yp) not in self.d.keys():
#           self.d[(y,yp)]=0
#       self.d[(y,yp)]+=1
#       return self.d[(y,yp)]
#     def __str__(self):
#         return self.d
    
class F1score: 
    """
    return the f1score for binary classes:
    can compute both the micro and macro f1score
    """
    def __init__(self,threshold = 0.5):
        self.threshold = threshold
        self.eps = 1e-6
        self.tp=[]
        self.fp=[]
        self.tn=[]
        self.fn=[]

    @torch.no_grad()
    def __call__(self, y_pr,y_gt):
        y_pr=y_pr>self.threshold
        self.tp.append(y_pr[y_gt].sum())
        self.fn.append((~y_pr)[y_gt].sum())
        self.fp.append(y_pr[~y_gt].sum())
        self.tn.append((~y_pr)[~y_gt].sum())

    def get_tp(self):
        return np.array(self.tp)
    def get_tn(self):
        return np.array(self.tn)
    def get_fp(self):
        return np.array(self.fp)
    def get_fn(self):
        return np.array(self.fn)
    def f1(self,tp,fp,fn):
        return 2*tp/(2*tp+fp+fn)
    
    def Macro_f1(self):
        return self.f1(self.get_tp().sum(),self.get_fp().sum(),self.get_fn().sum())
    def Micro_f1(self):
        return np.array(list(map(self.f1,self.tp,self.fp,self.fn))).mean()
    
def loss_and_metrics_single_class(y,yp,criterion =nn.BCEWithLogitsLoss(),threshold=0):

    """computes some metrics for comprison:
    returns:
    1.loss
    2.f1score
    3.overall accuracy
    4.user's accuracy
    5.producer's accuracy
    6.kappa score
    7.matthews correlation coefficient 
    """

    loss=criterion(torch.tensor(yp),torch.tensor(y)).item()
    yp=yp>=threshold
    acc=(yp==y).sum()/np.ones_like(y).sum() 
    user_acc=((yp*y).sum()+1e-5)/(y.sum()+1e-5) 
    prod_acc=((yp*y).sum()+1e-5)/(yp.sum()+1e-5) 
    kappa=metrics.cohen_kappa_score(yp,y)
    f1score=metrics.f1_score(y,yp)
    mc=metrics.matthews_corrcoef(y, yp)
    return loss,f1score,acc,user_acc,prod_acc,kappa,mc


def set_seed(seed=911):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_and_metrics_multi_class(y,yp,criterion =nn.CrossEntropyLoss(),threshold=0.5):
    loss=criterion(torch.tensor(yp),torch.tensor(y,dtype=torch.long)).item()#.softmax(axis=2)

    
    tp, fp, fn, tn = get_stats(torch.tensor(yp).argmax(1), torch.tensor(y,dtype=torch.long), mode='multiclass',num_classes=27)

    iou_score_res = float(iou_score(tp, fp, fn, tn, reduction="micro").numpy())
    f1_score_res = float(f1_score(tp, fp, fn, tn, reduction="micro").numpy())

    # yp=yp>=threshold
    yp=yp.argmax(axis=1)
    acc=(yp==y).sum()/np.ones_like(y).sum() 

    

    # user_acc=((yp==y).sum()+1e-5)/(y.sum()+1e-5) #check later
    # prod_acc=((yp==y).sum()+1e-5)/(yp.sum()+1e-5) #check later
    # kappa=metrics.cohen_kappa_score(yp,y)
    # f1score=metrics.f1_score(y,yp)
    # mc=metrics.matthews_corrcoef(y, yp)
    return loss,acc,iou_score_res,f1_score_res#,user_acc,prod_acc#,f1score,kappa,mc
