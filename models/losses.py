"""
File for auxiliary loss functions
"""
import torch

def ood_loss():
    pass

def calibration_loss(preds, mean=0, var=1):
    """ moment matching to standard normal """
    mean_loss = torch.mean(preds) -