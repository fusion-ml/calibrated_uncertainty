import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from utils import calibrate as cali

def train_plot(train_set, model):
    with torch.no_grad():
        full_train_gen = DataLoader(train_set, batch_size=len(train_set))
        full_train_data = next(iter(full_train_gen))[0].float()
        # order = full_train_data.argsort()[::-1]
        # full_train_sorted = full_train_data[order]
        full_train_pred = model(full_train_data)
        full_train_np = full_train_data.numpy()
        order = full_train_np.flatten().argsort()
        full_train_np_sorted = full_train_np[order]
        full_train_pred_sorted = full_train_pred.numpy()[order]

        plt.clf()
        plt.plot(full_train_np_sorted, full_train_pred_sorted, '-o')
        plt.show()
        import pdb; pdb.set_trace()


def ens_plot_all(train_set, test_gen, model_ens):
    test_residuals = []
    with torch.no_grad():
        for data in test_gen:
            test_X, test_y = data
            test_X, test_y = test_X.float(), test_y.float()

            test_X_flat = test_X.flatten()
            test_y_flat = test_y.flatten()
            test_order = test_X_flat.argsort().numpy()

            # import pdb; pdb.set_trace()
            pred_list = [ens_member(test_X).numpy() for ens_member in model_ens]
            preds = np.hstack(pred_list)
            pred_mean = np.mean(preds, axis=1)
            pred_std = np.std(preds, axis=1)

            test_residuals = test_y.numpy().reshape(pred_mean.shape) \
                             - pred_mean

            """ plot each member's prediction """
            plt.figure(figsize=(5, 4))
            for single_pred in pred_list:
                pass
                # plt.plot(test_X.numpy().flatten()[test_order],
                #          single_pred.flatten()[test_order],
                #          c='k', linewidth=0.3)
                # plt.scatter(test_X.numpy().flatten()[test_order],
                #          single_pred.flatten()[test_order],
                #          c='k', s=1)
            """ plot mean and stddev of prediction"""
            plt.errorbar(test_X.numpy().flatten()[test_order],
                         pred_mean.flatten()[test_order],
                         yerr=pred_std.flatten()[test_order], label='Mean Ensemble Prediction', color='r')
            """ plot ground truth """
            # plt.scatter(test_X.numpy().flatten()[test_order],
            #          test_y.numpy().flatten()[test_order], s=1, label='GT')
            plt.plot(test_X.numpy().flatten()[test_order],
                     train_set.oneD_fn(test_X.numpy().flatten()[test_order].reshape(-1, 1)), label='Ground Truth', color='b')
        # plt.axvline(args.train_min, c='k')
        # plt.axvline(args.train_max, c='k')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        # plt.ylim(-5, 1.5)
        plt.savefig('./figs/pred_plot.png')
        plt.show()

    """ test calibration """
    import pdb;
    pdb.set_trace()
    print(np.mean(test_residuals))
    print(pred_std)

    test_residuals = np.array(test_residuals).reshape(-1, 1)
    test_pred_stds = np.array(pred_std).reshape(-1, 1)
    exp, obs = cali.get_proportion_lists(test_residuals, test_pred_stds)
    cali.plot_calibration_curve(exp, obs, 'test')
    plt.savefig('./figs/calibration.png')
    plt.show()

    # import pdb;
    # pdb.set_trace()
    # plt.clf()
    # plt.hist(test_residuals / test_pred_stds)
    # plt.ylabel('Proportion in test set')
    # plt.xlabel('Standardized Residuals')
    # plt.legend()
    # plt.savefig('./figs/resid_prop.png')
    # plt.show()
    #
    # import pdb;
    # pdb.set_trace()
    # plt.clf()
    # plt.hist(test_pred_stds)
    # plt.ylabel('Proportions')
    # plt.xlabel('Predicted Test Standard Deviations')
    # plt.legend()
    # plt.savefig('./figs/stddev_prop.png')
    # plt.show()

    import pdb;
    pdb.set_trace()
    plt.clf()
    plt.scatter(test_X.numpy()[test_order], test_y.numpy()[test_order], s=2, color='b')
    plt.errorbar(test_X.numpy()[test_order],
                 pred_mean[test_order],
                 yerr=test_pred_stds[test_order], color='r')
    plt.legend()
    plt.show()

    import pdb;
    pdb.set_trace()
    plt.clf()
    plt.scatter(test_X.numpy()[test_order], test_y.numpy()[test_order],
                s=3)
    plt.plot(test_X.numpy()[test_order], pred_mean[test_order], label='Predictive Mean', color='r')
    plt.fill_between(test_X.numpy()[test_order].flatten(),
                     pred_mean[test_order].flatten() - 2 * test_pred_stds[test_order].flatten(),
                     pred_mean[test_order].flatten() + 2 * test_pred_stds[test_order].flatten(),
                     alpha=0.3, label='2*Predictive Stddev', color='r')
    plt.plot(test_X.numpy().flatten()[test_order],
             train_set.oneD_fn(test_X.numpy().flatten()[test_order].reshape(-1, 1)), label='Ground Truth Mean', color='b')
    plt.legend()
    plt.savefig('./figs/shaded.png')
    plt.show()

    import pdb;
    pdb.set_trace()