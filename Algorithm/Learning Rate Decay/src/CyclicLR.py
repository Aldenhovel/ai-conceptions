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


def plot_cyclic():
    plt.clf()
    optim = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.CyclicLR(
        optim, base_lr=0.1, max_lr=1., step_size_up=20, 
        mode='triangular', cycle_momentum=False)
    lrs = get_lr_scheduler(optim, scheduler, total_step)
    plt.plot(lrs, label='triangular')

    scheduler = lr_scheduler.CyclicLR(
        optim, base_lr=0.1, max_lr=1., step_size_up=20, 
        mode='triangular2', cycle_momentum=False)
    lrs = get_lr_scheduler(optim, scheduler, total_step)
    plt.plot(lrs, label='triangular2')

    scheduler = lr_scheduler.CyclicLR(
        optim, base_lr=0.1, max_lr=1., step_size_up=20, gamma=0.99,
        mode='exp_range', cycle_momentum=False)
    lrs = get_lr_scheduler(optim, scheduler, total_step)
    plt.plot(lrs, label='exp_range')
    plt.legend()
    plt.show()