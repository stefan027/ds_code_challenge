"""Metrics for binary classification using FastAI."""

import torch
from fastai.vision.all import (
    Metric,
    BalancedAccuracy,
    APScoreBinary,
    Precision,
    Recall,
    RocAucBinary
)


class BinaryClassificationMetric(Metric):
    def __init__(self, func, thres=None):
        self.func = func
        self.thres = thres

    def reset(self):
        self.preds, self.targs = [], []

    def accumulate(self, learn):
        preds_batch = learn.pred.detach().cpu().flatten()
        if self.thres is not None:
            preds_batch = (preds_batch > self.thres).float()
        preds_batch = preds_batch.numpy().tolist()
        targs_batch = learn.yb[0].detach().cpu().flatten().numpy().tolist()
        self.preds += preds_batch
        self.targs += targs_batch

    @property
    def value(self):
        if self.preds:
            return self.func(torch.tensor(self.preds), torch.tensor(self.targs))
        return None

    @property
    def name(self):
        return self.func.name


def balanced_accuracy(thres=0.5):
    return BinaryClassificationMetric(func=BalancedAccuracy(), thres=thres)

def ap_score():
    return BinaryClassificationMetric(func=APScoreBinary())

def precision(thres=0.5):
    return BinaryClassificationMetric(func=Precision(), thres=thres)

def recall(thres=0.5):
    return BinaryClassificationMetric(func=Recall(), thres=thres)

def roc_auc():
    return BinaryClassificationMetric(func=RocAucBinary())
