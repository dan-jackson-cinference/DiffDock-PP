from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def compute_rmsd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = ((x - y) ** 2).sum(-1)
    dist = dist / len(dist)  # normalize
    dist = dist.sum().sqrt()
    return dist


def compute_metrics(true, pred):
    """
    this function needs to be overhauled

    these lists are JAGGED IFF as_sequence=True
    @param pred (n, sequence, 1) preds for prob(in) where in = 1
    @param true (n, sequence, 1) targets, binary vector
    """
    # metrics depend on task
    as_sequence = type(true[0]) is list
    if as_sequence:
        f_metrics = {"roc_auc": _compute_roc_auc, "prc_auc": _compute_prc_auc}
    else:
        as_classification = (
            type(true[0]) == torch.Tensor and true[0].dtype == torch.long
        )
        if as_classification:
            f_metrics = {"topk_accuracy": _compute_topk}
        else:
            f_metrics = {
                "mse": _compute_mse,
            }
    scores = defaultdict(list)
    for key, f in f_metrics.items():
        if as_sequence:
            for t, p in zip(true, pred):
                scores[key].append(f(t, p))
            scores[key] = np.mean(scores[key])
        else:
            if as_classification:
                topk = f(true, pred)
                ks = [1, 5, 10]
                for i, val in enumerate(topk):
                    scores[f"{key}_{ks[i]}"] = val
            else:
                scores[key] = f(true, pred)
    return scores


def _compute_roc_auc(true, pred):
    try:
        return metrics.roc_auc_score(true, pred)
    except:
        # single target value
        return 0.5


def _compute_prc_auc(true, pred):
    if true.sum() == 0:
        return 0.5
    precision, recall, _ = metrics.precision_recall_curve(true, pred)
    prc_auc = metrics.auc(recall, precision)
    return prc_auc


def _compute_mse(true, pred):
    # technically order doesn't matter but "input" then "target"
    true, pred = torch.tensor(true), torch.tensor(pred)
    return F.mse_loss(pred, true).item()


def _compute_topk(true, pred, topk=[1, 5, 10]):
    """
    @param (list)  topk
    """
    if type(true) is list:
        true, pred = torch.stack(true), torch.stack(pred)
    true, pred = true.cpu().numpy(), pred.cpu().numpy()
    labels = np.arange(pred.shape[-1])
    topk_accs = []
    for k in topk:
        # NOTE this does not handle duplicates.
        # "correct" predictions are sorted by index
        acc = metrics.top_k_accuracy_score(true, pred, k=k, labels=labels)
        topk_accs.append(acc)
    return topk_accs


# if __name__ == "__main__":
#     # test topk
#     true = torch.arange(5)
#     pred = torch.eye(5)
#     topk = _compute_topk(true, pred, topk=[1, 3, 5])
#     print(topk)

#     true = torch.arange(4) + 1
#     pred = torch.eye(5)[:4]
#     topk = _compute_topk(true, pred, topk=[1, 3, 4])
#     print(topk)
