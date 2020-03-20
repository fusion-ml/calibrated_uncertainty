"""
Sparse ensemble in which:
- sparse weights are chosen by either the most sensitive layer
  or N most sensitive layers (N is the number of ensemble members
- ensemble loss can be just MSE or MSE+sharp+cali
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init

import numpy as np
import sys, copy, random
from collections import deque

sys.path.append('../')
from utils.args import parse_args
from utils.sensitivity import weight_sensitivity_analysis
from models.losses import calibration_loss, sharpness_loss
from utils.test_plots import ens_plot_all

LINESKIP = "="*10+'\n'


def train(args, device):
    use_bias = bool(args.bias)
    use_cali = bool(args.cali)
    use_sharp = bool(args.sharp)

    ###BEGIN: set custom x
    custom_x = np.concatenate([np.linspace(-5, -3, args.train_size//2),
                              np.linspace(3, 5, args.train_size // 2)])
    ###END: set custom x
    """ set data """
    train_set = args.dataset_method(x_min=args.train_min,
                                    x_max=args.train_max,
                                    size=args.train_size,
                                    distr=args.train_distr,
                                    mean=args.train_mu,
                                    std=args.train_sigma,
                                    noise=bool(args.noise),
                                    custom_x=custom_x)
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
    parent_model = args.model(bias=use_bias,
                              num_layers=args.num_layers,
                              hidden_size=args.hidden).float()

    """ set optimizer and loss """
    criterion = parent_model.loss
    parent_optimizer = optim.Adam(parent_model.parameters(), lr=args.lr)
    # optimizers = [optim.Adam(ens_member.parameters(), lr=args.lr)
    #               for ens_member in model_ens]

    """ begin training parent """
    running_loss = deque(maxlen=10)
    parent_lr_thresh = False
    for epoch in range(args.parent_ep):
        for batch_idx, batch_data in enumerate(train_gen):
            batch_X, batch_y = batch_data
            batch_X, batch_y = batch_X.float(), batch_y.float()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            parent_optimizer.zero_grad()

            batch_pred = parent_model(batch_X)
            loss = criterion(batch_pred, batch_y)
            running_loss.append(loss.detach().item())
            loss.backward()
            parent_optimizer.step()

        if epoch % 25 == 0:
            print('Epoch {}: running loss {}'.format(epoch, np.mean(running_loss)))
            avg_running_loss = np.mean(running_loss)
            """ lr decay """
            if (avg_running_loss < 0.5) and (not parent_lr_thresh):
                print('Setting parent lr thresh 1')
                parent_lr_thresh = True
                for param_group in parent_optimizer.param_groups:
                    param_group['lr'] = 0.005
            """ breaking condition """
            if avg_running_loss < 1e-3:
                break
    import pdb; pdb.set_trace()


    """ determine most sensitive layer """
    full_train_gen = DataLoader(train_set, batch_size=len(train_set))
    sens_data = next(iter(full_train_gen))[0].float()
    layer_sensitivities = weight_sensitivity_analysis(parent_model, sens_data)
    ### either choose as many layers as ensembles to change
    # layer_sens_idx = np.array(layer_sensitivities).argsort()[::-1][:args.num_ens]
    ### or choose just the most sensitive layer
    layer_sens_idx = np.repeat(np.argmax(layer_sensitivities), args.num_ens)

    """ set each ensemble member """
    num_ens = args.num_ens
    model_ens = []
    optimizers = []

    for ens_idx in range(num_ens):
        """ copy parent to child """
        child_model = copy.deepcopy(parent_model)

        """ set layer to unfreeze """
        unfreeze_idx = layer_sens_idx[ens_idx]
        """ freeze all the weights of child"""
        for param in child_model.parameters():
            param.requires_grad = False

        child_params = list(child_model.parameters())
        child_params[unfreeze_idx].requires_grad = True

        print(child_params[unfreeze_idx].size())
        if (len(list(child_params[unfreeze_idx].size()))) < 2:
            init.constant_(child_params[unfreeze_idx], 0.0)
        else:
            init.xavier_uniform_(child_params[unfreeze_idx], gain=1.0)
        for x in child_model.fcs.parameters():
            print(x.requires_grad)
        print(LINESKIP)
        model_ens.append(child_model)
        optimizers.append(optim.Adam(child_model.parameters(), lr=args.lr))
    print('Finished populating ensemble')

    ### BEGIN: check that each sparse layer is initialized and different
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
    import pdb; pdb.set_trace()
    ### END: check

    """ train each ensemble member """
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
            for ens_idx in range(len(model_ens)):
                optimizers[ens_idx].zero_grad()
                batch_pred = model_ens[ens_idx](batch_X)
                ens_preds.append(batch_pred)
                pred_loss.append(
                    torch.unsqueeze(criterion(batch_pred, batch_y),
                                    dim=-1)
                )

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
                loss = loss + sharp_loss

            # loss = (
            #         torch.mean(concat_pred_loss)
            #         + cali_mean_loss
            #         + cali_std_loss
            #         + sharp_loss
            # )
            running_loss.append(loss.detach().item())
            loss.backward()
            for ens_idx in range(len(model_ens)):
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

    ### BEGIN: check that ens is sparse
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
    ### END: check

    """ testing and plotting """
    ens_plot_all(train_set, test_gen, model_ens)

    # """ testing """
    # plt.figure(figsize=(5,4))
    # with torch.no_grad():
    #     for data in test_gen:
    #         test_X, test_y = data
    #         test_X, test_y = test_X.float(), test_y.float()
    #
    #         pred_list = [ens_member(test_X).numpy() for ens_member in model_ens]
    #         preds = np.hstack(pred_list)
    #         pred_mean = np.mean(preds, axis=1)
    #         pred_std = np.std(preds, axis=1)
    #
    #         for single_pred in pred_list:
    #             plt.plot(test_X.numpy(), single_pred, c='k', linewidth=0.5)
    #         plt.errorbar(test_X.numpy(), pred_mean, yerr=pred_std, label='Sparse Ensemble')
    #         plt.plot(test_X.numpy(), test_y.numpy(), label='Ground Truth')
    #     # plt.axvline(args.train_min, c='k')
    #     # plt.axvline(args.train_max, c='k')
    #     plt.xlabel('X')
    #     plt.ylabel('y')
    #
    #     plt.legend()
    #     plt.savefig('figs/sparse_ens_{}.png'.format(args.dataset))
    #     plt.show()

def main():
    args, device = parse_args()

    """ set seeds """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """ train and test """
    train(args, device)


if __name__=="__main__":
    main()
