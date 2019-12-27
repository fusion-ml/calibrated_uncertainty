import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import sys, copy, random
sys.path.append('../')
from utils.args import parse_args

LINESKIP = "="*10+'\n'


def train(args, device):
    max_epochs = args.num_epoch
    use_bias = bool(args.bias)
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
    parent_model = args.model(bias=use_bias,
                              num_layers=args.num_layers,
                              hidden_size=args.hidden).float()
    # model_ens = [args.model(hidden_size=args.hidden).float()
    #              for _ in range(args.num_ens)]

    """ set optimizer and loss """
    criterion = parent_model.loss
    parent_optimizer = optim.Adam(parent_model.parameters(), lr=args.lr)
    # optimizers = [optim.Adam(ens_member.parameters(), lr=args.lr)
    #               for ens_member in model_ens]

    """ begin training parent """
    for epoch in range(max_epochs):
        for batch_idx, batch_data in enumerate(train_gen):
            batch_X, batch_y = batch_data
            batch_X, batch_y = batch_X.float(), batch_y.float()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            parent_optimizer.zero_grad()

            batch_pred = parent_model(batch_X)
            loss = criterion(batch_pred, batch_y)
            loss.backward()
            parent_optimizer.step()
            # print(loss.item())
        print('Epoch {} finished'.format(epoch))

    """ set each ensemble member """
    # parent_weights = parent_model.parameters()
    stride = 2 if use_bias else 1
    # stride = 1

    num_ens = int(args.num_layers * 2 / stride)
    model_ens = []
    optimizers = []

    for ens_idx in range(num_ens):
        """ copy parent to child """
        child_model = copy.deepcopy(parent_model)

        """ freeze all the weights of child"""
        for param in child_model.parameters():
            param.requires_grad = False

        child_params = list(child_model.parameters())
        for layer_elem in range(stride):
            unfreeze_idx = (ens_idx*stride) + layer_elem
            child_params[unfreeze_idx].requires_grad = True
            # list(child_model.fcs)[unfreeze_idx//2].reset_parameters(reset_indv_bias=(unfreeze_idx%2==1))
        list(child_model.fcs)[ens_idx].reset_parameters()
        for x in child_model.fcs.parameters():
            print(x.requires_grad)
        print(LINESKIP)
        model_ens.append(child_model)
        optimizers.append(optim.Adam(child_model.parameters(), lr=args.lr))

    """ check that each sparse layer is initialized and different """
    print('Finished populating ensemble')
    for comp_ens_idx in range(num_ens-1):
        print('Comparing members {} and {}'.format(comp_ens_idx, comp_ens_idx+1))
        for param_idx in range(args.num_layers * 2):
            if (torch.sum(list(model_ens[comp_ens_idx].parameters())[param_idx]
                          - list(model_ens[comp_ens_idx+1].parameters())[param_idx])
                    == 0):
                print('param {} is the same'.format(param_idx))
            else:
                print('param {} is NOT the same'.format(param_idx))
        print(LINESKIP)

    """ train each ensemble member """
    for epoch in range(max_epochs):
        for batch_idx, batch_data in enumerate(train_gen):
            batch_X, batch_y = batch_data
            batch_X, batch_y = batch_X.float(), batch_y.float()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            for ens_idx in range(len(model_ens)):
                optimizers[ens_idx].zero_grad()

                batch_pred = model_ens[ens_idx](batch_X)
                loss = criterion(batch_pred, batch_y)
                loss.backward()
                optimizers[ens_idx].step()
            # print(loss.item())
        print('Epoch {} finished'.format(epoch))

    """ check that ens is sparse """
    print('Finished training each ensemble member')
    for comp_ens_idx in range(num_ens - 1):
        print('Comparing members {} and {}'.format(comp_ens_idx, comp_ens_idx+1))
        for param_idx in range(args.num_layers * 2):
            if (torch.sum(list(model_ens[comp_ens_idx].parameters())[param_idx]
                          - list(model_ens[comp_ens_idx + 1].parameters())[param_idx])
                    == 0):
                print('param {} is the same'.format(param_idx))
            else:
                print('param {} is NOT the same'.format(param_idx))
        print(LINESKIP)

    import pdb; pdb.set_trace()

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
                plt.plot(test_X.numpy(), single_pred, c='k', linewidth=0.1)
            plt.errorbar(test_X.numpy(), pred_mean, yerr=pred_std, label='preds')
            plt.plot(test_X.numpy(), test_y.numpy(), label='GT')
        plt.axvline(args.train_min, c='k')
        plt.axvline(args.train_max, c='k')
        plt.legend()
        plt.show()

def main():
    args, device = parse_args()

    """ set seeds """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args, device)



if __name__=="__main__":
    main()