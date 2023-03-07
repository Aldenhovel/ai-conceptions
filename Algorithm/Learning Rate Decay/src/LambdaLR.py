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


def plot_lambdalr():
    plt.clf()
    optim = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: step%100/100.)
    lrs = get_lr_scheduler(optim, scheduler, total_step)
    plt.plot(lrs)
    plt.title('LambdaLR')
    plt.show()