"""
Test script that sees if we can control the mean and spread of a vanilla ensemble:
- vanilla ensemble
- ensemble loss is distance of mean from 0, distance of std from 1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import sys, random
sys.path.append('../')
from utils.args import parse_args

LINESKIP = "="*10+'\n'

def train(args, device):
    max_epochs = args.num_epoch
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
    model_ens = [args.model(hidden_size=args.hidden).float()
                 for _ in range(args.num_ens)]

    """ set optimizer and loss """
    criterion = model_ens[0].loss
    optimizers = [optim.Adam(ens_member.parameters(), lr=args.lr)
                  for ens_member in model_ens]

    """ begin training """
    for epoch in range(max_epochs):
        for batch_idx, batch_data in enumerate(train_gen):
            batch_X, batch_y = batch_data
            batch_X, batch_y = batch_X.float(), batch_y.float()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            batch_preds = []
            for ens_idx in range(args.num_ens):
                optimizers[ens_idx].zero_grad()
                batch_pred = model_ens[ens_idx](batch_X)
                batch_preds.append(batch_pred)
            # import pdb; pdb.set_trace()
            concat_preds = torch.cat(batch_preds, dim=1)
            concat_means = torch.mean(concat_preds, dim=1)
            power_means = concat_means**2
            concat_stds = torch.std(concat_preds, dim=1)
            power_stds = (concat_stds - torch.ones_like(concat_stds))**2
            loss = torch.mean(power_stds) + torch.mean(power_means)
            if batch_idx%100==0:
                print(loss.item())
            loss.backward()
            for ens_idx in range(args.num_ens):
                optimizers[ens_idx].step()

        print('Epoch {} finished'.format(epoch))

    print(concat_preds)
    print(concat_means)
    print(concat_stds)
    """ testing """
    with torch.no_grad():
        for data in test_gen:
            test_X, test_y = data
            test_X, test_y = test_X.float(), test_y.float()

            import pdb; pdb.set_trace()
            pred_list = [ens_member(test_X).numpy() for ens_member in model_ens]
            preds = np.hstack(pred_list)
            pred_mean = np.mean(preds, axis=1)
            pred_std = np.std(preds, axis=1)

            for single_pred in pred_list:
                plt.plot(test_X.numpy(), single_pred, c='k', linewidth=0.5)
            plt.errorbar(test_X.numpy(), pred_mean, yerr=pred_std, label='preds')
            plt.plot(test_X.numpy(), test_y.numpy(), label='GT')
        plt.axvline(args.train_min, c='k')
        plt.axvline(args.train_max, c='k')
        plt.legend()
        plt.ylim(-5, 1.5)
        plt.show()

    print(LINESKIP)
    import pdb; pdb.set_trace()
    for ens_idx in range(len(model_ens)):
        torch.save(model_ens[ens_idx].state_dict(),
                   './saved_weights/model_{}.pt'.format(ens_idx))



def main():
    args, device = parse_args()

    """ set seeds """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args, device)



if __name__=="__main__":
    main()