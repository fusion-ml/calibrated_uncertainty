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

    import pdb; pdb.set_trace()
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

            # import pdb; pdb.set_trace()
            concat_pred_loss = torch.cat(pred_loss)
            concat_preds = torch.cat(ens_preds, dim=1)
            pred_stds = torch.std(concat_preds, dim=1)
            pred_means = torch.mean(concat_preds, dim=1)
            pred_resids = pred_means - torch.flatten(batch_y)

            resid_z_scores = pred_resids/pred_stds
            mean_z_scores = torch.mean(resid_z_scores)
            std_z_scores = torch.std(resid_z_scores)

            from torch.distributions import Normal
            std_norm = Normal(0.0, 1.0)
            cdf_labels = std_norm.cdf(resid_z_scores)

            resid_z_argsort = resid_z_scores.argsort()
            ecdf_pred = resid_z_argsort.float()/resid_z_argsort.max().float()
            ecdf_pred = Variable(ecdf_pred.data-resid_z_scores.data) + resid_z_scores
            loss =
            """ calculate losses """
            mean_z_score_loss = mean_z_scores**2
            std_z_score_loss = (std_z_scores - 1)**2

            loss = (
                    torch.mean(concat_pred_loss)   # prediction loss (MSE)
                    + torch.mean(pred_stds)        # sharpness
                    + std_z_score_loss             # make std = 1
                    + mean_z_score_loss            # make mean = 0
            )
                   # + mean_z_score_loss \
                   # + std_z_score_loss
            # loss = torch.mean(mean_z_score_loss) \
            #        + torch.mean(std_z_score_loss)

            """            
            # pred loss
            # concat_mse = torch.cat(pred_loss, dim=-1)
            #
            # mean loss
            concat_resids = torch.cat(residuals, dim=1)

            # concat_resid_means = torch.mean(concat_resids, dim=1)
            # power_means = concat_resid_means**2

            # std loss
            # concat_stds = torch.std(concat_resids, dim=1)
            # power_stds = (concat_stds - torch.ones_like(concat_stds))**2

            concat_stds = torch.std(concat_resids, dim=1).reshape(-1,1)
            concat_z_scores = concat_resids/concat_stds
            z_score_mean_loss = concat_z_scores**2
            z_score_std_loss = (concat_z_scores - torch.ones_like(concat_stds))**2
            # power_stds = (concat_stds - torch.ones_like(concat_stds)) ** 2
            """

            """ total loss """
            # loss = torch.mean(concat_mse) \
            #        + torch.mean(power_z_scores) \
            #        + torch.mean(power_means) \
            # loss = torch.mean(concat_resids**2) \
            #        + torch.mean(z_score_std_loss) \

                   # + torch.mean(concat_z_scores) \
                # + torch.mean(power_stds) \

            loss.backward()
            for ens_idx in range(args.num_ens):
                optimizers[ens_idx].step()
        if epoch % 25 == 0:
            print(loss.item())

        # print('Epoch {} finished'.format(epoch))

    print(resid_z_scores)
    print(mean_z_scores)
    print(std_z_scores)


    """ testing """
    test_pred_means = []
    test_residuals = []
    test_pred_stds = []
    with torch.no_grad():
        for data in test_gen:
            test_X, test_y = data
            test_X, test_y = test_X.float(), test_y.float()

            test_X_flat = test_X.flatten()
            test_y_flat = test_y.flatten()
            test_order = test_X_flat.argsort().numpy()

            import pdb; pdb.set_trace()
            pred_list = [ens_member(test_X).numpy() for ens_member in model_ens]
            preds = np.hstack(pred_list)
            pred_mean = np.mean(preds, axis=1)
            pred_std = np.std(preds, axis=1)

            # test_pred_means = pred_mean
            test_residuals = test_y.numpy().reshape(pred_mean.shape)\
                             - pred_mean
            # test_pred_stds = pred_std

            import pdb; pdb.set_trace()
            """ plot each member's prediction """
            for single_pred in pred_list:
                plt.plot(test_X.numpy().flatten()[test_order],
                         single_pred.flatten()[test_order],
                         c='k', linewidth=0.5)
            """ plot mean and stddev of prediction"""
            plt.errorbar(test_X.numpy().flatten()[test_order],
                         pred_mean.flatten()[test_order],
                         yerr=pred_std.flatten()[test_order], label='preds')
            """ plot ground truth """
            plt.plot(test_X.numpy().flatten()[test_order],
                     test_y.numpy().flatten()[test_order], label='GT')
        plt.axvline(args.train_min, c='k')
        plt.axvline(args.train_max, c='k')
        plt.legend()
        # plt.ylim(-5, 1.5)
        plt.show()

    """ test calibration """
    import pdb; pdb.set_trace()
    print(np.mean(test_residuals))
    print(pred_std)

    test_residuals = np.array(test_residuals).reshape(-1,1)
    test_pred_stds = np.array(pred_std).reshape(-1,1)
    exp, obs = cali.get_proportion_lists(test_residuals, test_pred_stds)
    cali.plot_calibration_curve(exp, obs, 'test')
    plt.show()

    import pdb; pdb.set_trace()
    plt.clf()
    plt.hist(test_residuals/test_pred_stds)
    plt.show()

    import pdb; pdb.set_trace()
    plt.clf()
    plt.plot(test_X.numpy()[test_order], test_y.numpy()[test_order])
    plt.errorbar(test_X.numpy()[test_order],
                 pred_mean[test_order],
                 yerr=test_pred_stds[test_order])
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