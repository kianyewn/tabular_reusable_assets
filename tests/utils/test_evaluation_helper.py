import numpy as np
import torch

from tabular_reusable_assets.utils.evaluation_helper import recall


def test_recall():
    preds = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]])
    hits = torch.tensor([[1, 0, 0, 1, 1], [1, 0, 0, 1, 1], [1, 0, 0, 1, 1]])
    assert np.allclose(recall(preds, hits, k=2).item(), 0.33333)
