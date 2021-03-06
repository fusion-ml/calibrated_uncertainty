"""
Single model:
  train and test either a single deterministic network
  or a probabilistic network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from _collections import deque
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from models.model import *
from utils.args import parse_args
from utils.test_plots import train_plot

def train(args, device):
    """ set data """
    train_set = args.dataset_method(x_min=args.train_min,
                                    x_max=args.train_max,
                                    size=args.train_size,
                                    distr=args.train_distr,
                                    mean=args.train_mu,
                                    std=args.train_sigma)
    train_gen = DataLoader(train_set, **args.train_data_params)

    test_set = args.dataset_method(x_min=args.test_min,
                                   x_max=args.test_max,
                                   size=args.test_size,
                                   distr=args.test_distr,
                                   mean=args.test_mu,
                                   std=args.test_sigma)
    test_gen = DataLoader(test_set, **args.test_data_params)

    """ set model """
    model = args.model(input_size=1, hidden_size=args.hidden)
    model = model.float()

    """ set optimizer and loss """
    criterion = model.loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    """ begin training """
    running_loss = deque(maxlen=10)
    for epoch in range(args.parent_ep):
        for batch_idx, batch_data in enumerate(train_gen):
            batch_X, batch_y = batch_data
            batch_X, batch_y = batch_X.float(), batch_y.float()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            batch_pred = model(batch_X)
            loss = criterion(batch_pred, batch_y)
            running_loss.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        if epoch % 25 == 0:
            print('Epoch {}: running loss {}'.format(epoch, np.mean(running_loss)))
            avg_running_loss = np.mean(running_loss)
            """ lr decay """
            if avg_running_loss < 0.5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.005
            """ breaking condition """
            if avg_running_loss < 1e-3:
                break
    import pdb; pdb.set_trace()

    """ plot result of training """
    train_plot(train_set, model)

    """ testing """
    with torch.no_grad():
        for data in test_gen:
            test_X, test_y = data
            test_X, test_y = test_X.float(), test_y.float()

            if args.model_type == 'vanilla':
                pred = model(test_X)
                plt.plot(test_X.numpy(), pred.numpy())
            elif args.model_type == 'pnn':
                pred = model(test_X)

                pred_mean = pred[:, :model.mean_dim].numpy()
                pred_var = pred[:, model.mean_dim].numpy()
                pred_std = np.std(pred_var)
                plt.errorbar(test_X.numpy(), y=pred_mean, yerr=pred_std)
            plt.plot(test_X.numpy(), test_y.numpy())
        plt.show()

def main():
    args, device = parse_args()
    train(args, device)


if __name__=="__main__":
    main()
