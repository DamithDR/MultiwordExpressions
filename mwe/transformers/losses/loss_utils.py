import torch
import warnings
from torch.nn import CrossEntropyLoss

from mwe.transformers.losses.dice_loss import DiceLoss
from mwe.transformers.losses.focal_loss import FocalLoss
from mwe.transformers.losses.tversky_loss import TverskyLoss


def init_loss(weight, device, args):
    if weight and args.loss_type:
        warnings.warn(
            f"weight and args.loss_type parametters are set at the same time"
            f"will use weighted cross entropy loss. To use {args.loss_type} set weight to None"
        )
    if weight:
        loss_fct = CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
    elif args.loss_type:
        if args.loss_type == "focal":
            loss_fct = FocalLoss(**args.loss_args)
        elif args.loss_type == "dice":
            loss_fct = DiceLoss(**args.loss_args)
        elif args.loss_type == "tversky":
            loss_fct = TverskyLoss(**args.loss_args)
        else:
            raise NotImplementedError(f"unknown {args.loss_type} loss function")
    else:
        loss_fct = None

    return loss_fct
