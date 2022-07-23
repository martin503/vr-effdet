import torch
import random
import numpy as np

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_model(model, optim, loss_mode, epoch, lr, metric):
    name = optim.__class__.__name__ + '_' + loss_mode + '_e-' + str(epoch) + '_lr-' + str(int(10000 * lr)) + '_m-' + str(int(100 * metric))
    torch.save(model, './models/' + name + '.ckpt')
    return name

def discretize(x, threshold):
    return (x > threshold).type(torch.float32)
