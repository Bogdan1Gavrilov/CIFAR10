import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnWarmupRestart(_LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=1e-6,
                 warmup_steps=0,
                 gamma=1.0, #Коэффициент уменьшения max_lr посл екаждого цикла
                 last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.optimizer.param_groups]
            
        if self.step_in_cycle < self.warmup_steps:
            #происходит линейный разогрев lr(быстро растем)
            return [self.min_lr + (self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps
                    for _ in self.optimizer.param_groups]
        else:
            #происходит косинусное затухание lr
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr +(self.max_lr * (self.gamma ** self.cycle) - self.min_lr) * cosine_decay
            return [lr for _ in self.optimizer.param_groups]
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >=  self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = int(self.first_cycle_steps * self.cycle_mult ** n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                self.cycle = 0

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr