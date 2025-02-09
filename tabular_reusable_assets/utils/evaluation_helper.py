from typing import List

import torch
from sklearn.metrics import get_scorer, roc_auc_score, roc_curve


class Scorer:
    def __init__(self, scoring: List):
        self.scoring = scoring
        self.create_scorers()

    def create_scorers(self):
        self.scorers = {}
        for metric in self.scoring:
            self.scorers[metric] = get_scorer(metric)
        return

    def __call__(self, model, X, y):
        scores = {}
        for metric, scorer in self.scorers.items():
            scores[metric] = scorer(model, X, y)
        return scores


def monotonicity(data, feat, target_column):
    """if data is non linear, converting it to bins can and make it monotonic"""
    # AUC-ROC only captures monotonic relationships (linear relationship); ability to discriminate between two classes
    # low auc: feature values are randomly distributed across both classes
    # high auc: higher/lower value features increases or decreases the target label rates
    auc = roc_auc_score(y_true=data[target_column], y_score=data[feat])
    fpr, tpr, _ = roc_curve(y_true=data[target_column], y_score=data[feat])
    ks = max(abs(tpr - fpr))
    return auc, ks


def recall(preds: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """calculate recall

    Args:
        preds (torch.Tensor): shape (batch_size, n_items)
        target (torch.Tensor): shape (batch_size, n_items)

    Returns:
        float: _description_
    """
    n_relevant_items = target.sum(dim=1)
    hits = target[:, :k].sum(dim=1)
    batch_recall = hits / n_relevant_items
    # print(hits.shape, n_relevant_items.shape)
    return batch_recall.sum() / preds.shape[0]
