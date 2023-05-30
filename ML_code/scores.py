import numpy as np
import torch

def accuracy(y_hat, y_true):
    return torch.sum(((y_hat == y_true)) / len(y_hat)).item()

def TP_FP_TN_FN(y_hat, y_true, label_to_calc):
    y_hat = np.array([1 if yh == label_to_calc else 0 for yh in y_hat ])
    y_true = np.array([1 if yt == label_to_calc else 0 for yt in y_true])

    tp = np.sum((y_hat * y_true))
    fp = np.sum(np.array([1 for yh, yt in zip(y_hat, y_true) if yh and not yt]))
    tn = np.sum(np.array([1 for yh, yt in zip(y_hat, y_true) if not yh and not yt]))
    fn = np.sum(np.array([1 for yh, yt in zip(y_hat, y_true) if not yh and yt]))
    return tp, fp, tn, fn
    
def precision(true_positives, false_positives):
    if true_positives+false_positives == 0:
        return 0

    return true_positives / (true_positives+false_positives)

def recall(true_positives, false_negatives):

    if true_positives+false_negatives == 0:
        return 0

    return true_positives / (true_positives+false_negatives)

def f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0

    return 2*((precision * recall) / (precision + recall))

def kappa_score(y_hat, y_true):

    N = len(y_hat)
    p0 = torch.sum(((y_hat == y_true)) / N).item()
    pe = 0
    for label in y_true.unique():
        y_hat_label = np.array([1 if yh == label else 0 for yh in y_hat ])
        y_true_label = np.array([1 if yt == label else 0 for yt in y_true])
        pe_yhat = np.sum(np.array(y_hat_label)) / N
        pe_ytrue = np.sum(np.array(y_true_label)) / N
        pe += pe_yhat * pe_ytrue

    return (p0 - pe) / (1 - pe)



