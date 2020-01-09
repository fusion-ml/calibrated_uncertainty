"""
File for auxiliary loss functions
"""
import torch

def ood_loss():
    pass

def calibration_loss(ens_preds, batch_y):
    """ moment matching to standard normal """
    concat_preds = torch.cat(ens_preds, dim=1)
    pred_stds = torch.std(concat_preds, dim=1)
    pred_means = torch.mean(concat_preds, dim=1)
    pred_resids = pred_means - torch.flatten(batch_y)

    resid_z_scores = pred_resids / pred_stds
    mean_z_scores = torch.mean(resid_z_scores)
    std_z_scores = torch.std(resid_z_scores)

    """ calculate losses """
    mean_z_score_loss = mean_z_scores ** 2
    std_z_score_loss = (std_z_scores - 1) ** 2

    return mean_z_score_loss, std_z_score_loss

def sharpness_loss(ens_preds):
    concat_preds = torch.cat(ens_preds, dim=1)
    pred_stds = torch.std(concat_preds, dim=1)

    sharp_loss = torch.mean(pred_stds)

    return sharp_loss