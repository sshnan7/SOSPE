import copy

import torch
import torch.nn as nn


class EMA_teacher(nn.Module):
    """
    Modified version of class fairseq.models.ema.EMAModule.

    Args:
        model (nn.Module):
        cfg (DictConfig):
        device (str):
        skip_keys (list): The keys to skip assigning averaged weights to.
    """

    def __init__(self, model):
        super().__init__()
        self.model = self.deepcopy_model(model) #see def deepcopy_model
        self.model.requires_grad_(False)
        
        #device = self.model.device
        #self.model.to(device)
        
        self.decay = 0.9998
        self.num_updates = 0
    
    def deepcopy_model(self, model):
        model = copy.deepcopy(model)
        return model

    def step(self, new_model: nn.Module):
        """
        One EMA step

        Args:
            new_model (nn.Module): Online model to fetch new weights from

        """
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            ema_param.mul_(self.decay)
            ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay) #parameter calculate by EMA
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False) #change parameter
        self.num_updates += 1

    def forward(self, x) :
        with torch.no_grad():
            device = x.device
            self.model.eval()
            self.model.to(device)
            x = self.model(x, selfsupervised = False, attn_mean= True)
        
        return x

class selfsupervised_criterion(nn.Module) :
    def __init__(self, ):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(reduction='none', beta=2) #nn.MSELoss()
    
    def forward(self, target, pred, grad_masking) :
        target = torch.mul(target, grad_masking) #target*grad_masking
        pred = torch.mul(pred, grad_masking) # pred*grad_masking
        
        #loss = self.criterion(pred, target)
        loss = self.criterion(pred, target).sum(dim=-1).sum().div(pred.size(0))
        
        loss_dict = {
            'selfsupervised' : loss
        }
        
        return loss_dict, loss
        


def build_teachermodel(student_model):  
    
    teacher_model = EMA_teacher(student_model)
    
    return teacher_model

def build_selfsupervised_criterion() :
    return selfsupervised_criterion()