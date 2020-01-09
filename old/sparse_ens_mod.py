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
from utils.args import *

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
    print('Training dataset: {}, min {}, max {}'.format(len(train_set),
                                                        args.train_min,
                                                        args.train_max))

    test_set = args.dataset_method(x_min=args.test_min,
                                   x_max=args.test_max,
                                   size=args.test_size,
                                   distr=args.test_distr,
                                   mean=args.test_mu,
                                   std=args.test_sigma)
    test_gen = DataLoader(test_set, **args.test_data_params)
    print('Testing dataset: {}, min {}, max {}'.format(len(test_set),
                                                        args.test_min,
                                                        args.test_max))

    """ OOD dataset """
    OOD_size = 100
    custom_x = np.concatenate([np.linspace(args.train_min-3, args.train_min, OOD_size // 2),
                               np.linspace(args.train_max, args.train_max + 3, OOD_size // 2)])
    OOD_set = args.dataset_method(size=args.test_size,
                                  distr=args.test_distr,
                                  custom_x=custom_x,
                                  mean=args.test_mu,
                                  std=args.test_sigma)
    OOD_gen = DataLoader(OOD_set, **args.train_data_params)
    print('OOD dataset: {}, min {}, max {}'.format(len(OOD_set),
                                                       np.min(custom_x),
                                                       np.max(custom_x)))
    import pdb; pdb.set_trace()


    """ set model """
    parent_model = args.model(bias=use_bias,
                              num_layers=args.num_layers,
                              hidden_size=args.hidden).float()


    """ set optimizer and loss """
    criterion = parent_model.loss
    parent_optimizer = optim.Adam(parent_model.parameters(), lr=args.lr)

    """ begin training parent model """
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
    children_params = []

    for ens_idx in range(num_ens):
        """ copy parent to child """
        child_model = copy.deepcopy(parent_model)

        """ freeze all the weights of child"""
        for param in child_model.parameters():
            param.requires_grad = False

        child_params = list(child_model.parameters())
        for layer_elem in range(stride):
            unfreeze_idx = (ens_idx*stride) + layer_elem
            print(unfreeze_idx, ens_idx, stride, layer_elem)
            child_params[unfreeze_idx].requires_grad = True
            # list(child_model.fcs)[unfreeze_idx//2].reset_parameters(reset_indv_bias=(unfreeze_idx%2==1))
        list(child_model.fcs)[ens_idx].reset_parameters()
        for x in child_model.fcs.parameters():
            print(x.requires_grad)
        print(LINESKIP)
        """ add child to ens """
        model_ens.append(child_model)
        children_params.extend(list(child_model.parameters()))
    print('Finished populating ensemble')

    """ set optimizer for children """
    children_optimizer = optim.Adam(children_params, lr=args.lr)

    """ check that each sparse layer is initialized and different """
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
            # pdb.set_trace()
            batch_X, batch_y = batch_data
            batch_X, batch_y = (batch_X.float()).to(device), (batch_y.float()).to(device)

            OOD_data = next(iter(OOD_gen))
            OOD_X, OOD_y = OOD_data
            OOD_X, OOD_y = (OOD_X.float()).to(device), (OOD_y.float()).to(device)

            children_optimizer.zero_grad()

            in_loss_list = []
            OOD_pred_list = []
            for ens_idx in range(len(model_ens)):
                batch_pred = model_ens[ens_idx](batch_X)
                in_loss_list.append(criterion(batch_pred, batch_y))

                OOD_pred_list.append(model_ens[ens_idx](OOD_X))

            in_loss = torch.mean(torch.stack(in_loss_list))
            OOD_loss = torch.mean(torch.var(torch.cat(OOD_pred_list, dim=1), dim=1))

            loss = in_loss - args.ood*OOD_loss
            loss.backward()
            children_optimizer.step()
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