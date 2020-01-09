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
from utils import calibrate as cali
from models.losses import calibration_loss, sharpness_loss
from utils.test_plots import ens_plot_all

LINESKIP = "="*10+'\n'

def train(args, device):
    max_epochs = args.num_epoch
    """ set data """
    train_set = args.dataset_method(x_min=args.train_min,
                                    x_max=args.train_max,
                                    size=args.train_size,
                                    distr=args.train_distr,
                                    mean=args.train_mu,
                                    std=args.train_sigma,
                                    noise=bool(args.noise))
    train_gen = DataLoader(train_set, **args.train_data_params)

    test_set = args.dataset_method(x_min=args.test_min,
                                   x_max=args.test_max,
                                   size=args.test_size,
                                   distr=args.test_distr,
                                   mean=args.test_mu,
                                   std=args.test_sigma,
                                   noise=bool(args.noise))
    test_gen = DataLoader(test_set, **args.test_data_params)

    """ set model """
    model_ens = [args.model(hidden_size=args.hidden).float()
                 for _ in range(args.num_ens)]

    """ set optimizer and loss """
    criterion = model_ens[0].loss
    optimizers = [optim.Adam(ens_member.parameters(), lr=args.lr)
                  for ens_member in model_ens]

    # import pdb; pdb.set_trace()
    """ begin training """
    for epoch in range(max_epochs):
        for batch_idx, batch_data in enumerate(train_gen):
            batch_X, batch_y = batch_data
            batch_X, batch_y = batch_X.float(), batch_y.float()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            pred_loss = []
            ens_preds = []
            for ens_idx in range(args.num_ens):
                optimizers[ens_idx].zero_grad()
                batch_pred = model_ens[ens_idx](batch_X)
                """ keep track of predictions, residuals, losses """
                ens_preds.append(batch_pred)
                pred_loss.append(
                    torch.unsqueeze(criterion(batch_pred, batch_y), dim=-1))

            """ criterion loss"""
            concat_pred_loss = torch.cat(pred_loss)
            """ calibration loss """
            cali_mean_loss, cali_std_loss = calibration_loss(ens_preds, batch_y)
            """ sharpness loss """
            sharp_loss = sharpness_loss(ens_preds)

            # print('DEBUG {}, {}, {}'.format(cali_mean_loss, cali_std_loss, sharp_loss))
            loss = (
                torch.mean(concat_pred_loss)
                + cali_mean_loss
                + cali_std_loss
                + sharp_loss
            )


            # concat_preds = torch.cat(ens_preds, dim=1)
            # pred_stds = torch.std(concat_preds, dim=1)
            # pred_means = torch.mean(concat_preds, dim=1)
            # pred_resids = pred_means - torch.flatten(batch_y)
            #
            # resid_z_scores = pred_resids/pred_stds
            # mean_z_scores = torch.mean(resid_z_scores)
            # std_z_scores = torch.std(resid_z_scores)
            #
            # """ calculate losses """
            # mean_z_score_loss = mean_z_scores**2
            # std_z_score_loss = (std_z_scores - 1)**2
            #
            # loss = (
            #         torch.mean(concat_pred_loss)   # prediction loss (MSE)
            #         + torch.mean(pred_stds)        # sharpness
            #         + std_z_score_loss             # make std = 1
            #         + mean_z_score_loss            # make mean = 0
            # )
            #
            # print('DEBUG {}, {}, {}'.format(mean_z_score_loss, std_z_score_loss, torch.mean(pred_stds)))

            loss.backward()
            for ens_idx in range(args.num_ens):
                optimizers[ens_idx].step()
        if epoch % 25 == 0:
            print(loss.item())

        # print('Epoch {} finished'.format(epoch))

    # print(resid_z_scores)
    # print(mean_z_scores)
    # print(std_z_scores)

    """ testing """
    ens_plot_all(train_set, test_gen, model_ens)

    """ save models """
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
