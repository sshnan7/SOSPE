import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, up_time = 1, gamma=1., alpha = 0.1, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0 #한 주기
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up #warm up 까지
        self.T_i = T_0 #한 주기
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch #T_cur 주기 내 현재 위치
        self.up_time = up_time #warmup 회수
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up: 
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1 #이전 epoch  +1
            self.T_cur = self.T_cur + 1 #이전 T_cur +1
            if self.T_cur >= self.T_up :
                if self.T_i-self.T_up == 0 :
                    self.cycle = self.cycle-1
                self.up_time = self.up_time -1
                if self.up_time <= 0 :
                    self.T_up = 0
            if self.T_cur >= self.T_i: #T_cur이 한 주기 + warmup 보다 크면?
                self.cycle += 1
                self.T_cur = self.T_cur - (self.T_i+self.T_up) #T_cur은 한 바퀴 내에서 현재 위치
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up #새로운 주기 정의 T_mult = 1이면 주기 유지
        else:
            if epoch >= self.T_0:#epoch이 한주기 보다 큰 경우
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0 #몇 주기 완료했나
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            #print(param_group['alpha'])
            param_group['lr'] = lr
            if param_group['alpha'] != 1 :
                param_group['lr'] = lr*param_group['alpha']