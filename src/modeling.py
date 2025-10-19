"""Modeling related classes and functions for the image classification task."""

import torch
from torch import nn
import timm
from fastai.vision.all import create_head, apply_init


def create_timm_model(
    arch, n_out, pretrained=True, n_in=3, init=nn.init.kaiming_normal_,
    custom_head=None, concat_pool=True, lin_ftrs=None, ps=0.5,
    first_bn=True, bn_final=False, lin_first=False, y_range=None,
    use_fastai_head=True, **kwargs
):
    """
    Modified from fastai.vision.learner.create_timm_model to correctly cut the head
    for Timm models.
    Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library
    """
    if use_fastai_head:
        model = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=n_in, **kwargs)
        nf = model.num_features
        pool = model.default_cfg.get('pool_size', None) is not None
        if custom_head is None:
            head = create_head(nf, n_out, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
                            first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range)
            if init is not None:
                apply_init(head, init)
            model.head = head
        else:
            model.head = custom_head
    else:
        model = timm.create_model(arch, pretrained=pretrained, num_classes=n_out, in_chans=n_in, **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), model.default_cfg


def freeze_except_head(model):
    """Freeze all model parameters except for those in the classification head."""
    for name, param in model.named_parameters():
        if name.split(".")[0] == "head":
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def unfreeze_all(model):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return model
