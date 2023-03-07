import os
import torch
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt


def get_lr_scheduler(optim, scheduler, total_step):
    '''
    get lr values
    '''
    lrs = []
    for step in range(total_step):
        lr_current = optim.param_groups[0]['lr']
        lrs.append(lr_current)
        if scheduler is not None:
            scheduler.step()
    return lrs
# global
model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3))
initial_lr = 1.
total_step = 200

def plot_cosine_aneal():
    plt.clf()
    optim = torch.optim.Adam([{'params': model.parameters(),
                            'initial_lr': initial_lr}], lr=initial_lr)
    # optim = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optim, T_max=40, eta_min=0.2)
    lrs = get_lr_scheduler(optim, scheduler, total_step)
    plt.plot(lrs, label='not use last epoch')
    # pdb.set_trace()
    
    # if not re defined, the init lr will be lrs[-1]
    optim = torch.optim.Adam([{'params': model.parameters(),
                            'initial_lr': initial_lr}], lr=initial_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optim, T_max=40, eta_min=0.2, last_epoch=10)
    lrs = get_lr_scheduler(optim, scheduler, total_step)
    plt.plot(lrs, label='last epoch=10')

    plt.title('CosineAnnealingLR')
    plt.legend()
    plt.show()