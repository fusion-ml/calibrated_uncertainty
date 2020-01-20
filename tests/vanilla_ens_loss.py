"""
Vanilla ensemble in which:
- ensemble loss can be just MSE or MSE+sharp+cali
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import sys, random
from collections import deque
sys.path.append('../')
from utils.args import parse_args
from models.losses import calibration_loss, sharpness_loss
from utils.test_plots import ens_plot_all

LINESKIP = "="*10+'\n'

def train(args, device):
    use_bias = bool(args.bias)
    use_cali = bool(args.cali)
    use_sharp = bool(args.sharp)
    # use_sharp = False

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
    model_ens = [args.model(bias=use_bias,
                            num_layers=args.num_layers,
                            hidden_size=args.hidden).float()
                 for _ in range(args.num_ens)]

    """ set optimizer and loss """
    criterion = model_ens[0].loss
    optimizers = [optim.Adam(ens_member.parameters(), lr=args.lr)
                  for ens_member in model_ens]

    # import pdb; pdb.set_trace()
    """ begin training """
    running_loss = deque(maxlen=10)
    lr_thresh_1 = False
    lr_thresh_2 = False
    lr_thresh_3 = False
    for epoch in range(args.ens_ep):
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

            """ criterion loss """
            concat_pred_loss = torch.cat(pred_loss)
            loss = torch.mean(concat_pred_loss)

            """ calibration loss """
            if use_cali:
                cali_mean_loss, cali_std_loss = calibration_loss(
                    ens_preds, batch_y)
                loss = loss + cali_mean_loss + cali_std_loss
            """ sharpness loss """
            if use_sharp:
                sharp_loss = sharpness_loss(ens_preds)
                loss = loss + 0.5*sharp_loss

            # """ criterion loss"""
            # concat_pred_loss = torch.cat(pred_loss)
            # """ calibration loss """
            # cali_mean_loss, cali_std_loss = calibration_loss(ens_preds, batch_y)
            # """ sharpness loss """
            # sharp_loss = sharpness_loss(ens_preds)
            #
            # # print('DEBUG {}, {}, {}'.format(cali_mean_loss, cali_std_loss, sharp_loss))
            # loss = (
            #     torch.mean(concat_pred_loss)
            #     + cali_mean_loss
            #     + cali_std_loss
            #     + sharp_loss
            # )


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
            running_loss.append(loss.detach().item())
            loss.backward()
            for ens_idx in range(args.num_ens):
                optimizers[ens_idx].step()

        if epoch % 25 == 0:
            print('Epoch {}: running loss {}'.format(epoch, np.mean(running_loss)))
            print('calis:{:.2f}, {:.2f}'.format(cali_mean_loss.item(),
cali_std_loss.item()))
            avg_running_loss = np.mean(running_loss)
            """ lr decay """
            if (avg_running_loss < 60) and (not lr_thresh_1):
                print('Setting lr thresh 1')
                for ens_idx in range(args.num_ens):
                    for param_group in optimizers[ens_idx].param_groups:
                        lr_thresh_1 = True
                        param_group['lr'] = 0.05
            if (avg_running_loss < 30) and (not lr_thresh_2):
                print('Setting lr thresh 2')
                for ens_idx in range(args.num_ens):
                    for param_group in optimizers[ens_idx].param_groups:
                        lr_thresh_2 = True
                        param_group['lr'] = 0.01
            if (avg_running_loss < 15) and (not lr_thresh_3):
                print('Setting lr thresh 3')
                for ens_idx in range(args.num_ens):
                    for param_group in optimizers[ens_idx].param_groups:
                        lr_thresh_3 = True
                        param_group['lr'] = 0.0005
            """ breaking condition """
            if (avg_running_loss < 10) \
                and (cali_mean_loss.item() < 0.03) \
                and (cali_std_loss.item() < 0.03):
                break
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
