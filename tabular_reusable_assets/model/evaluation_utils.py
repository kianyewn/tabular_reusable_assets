from typing import List

from sklearn.metrics import get_scorer


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
