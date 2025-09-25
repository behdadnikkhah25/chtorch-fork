# Lag en ny klasse EnsembleEstimator
# med samme interface som Estimator. Inni train må du trene og evalere forskjellige configurereinger av estimator og lagre scores
#I predict må du predicte med hver av configured estimators. Og returnere en weighted average av prediksjoner
# Lag en test som kjoerer nye klassen slik at du kan bruke debuggern
# EnsembleEstimator boer ligger i ensemble_estimator.py på samme nivaa som estimator.py
# Testing

from chtorch.estimator import Estimator
import numpy as np

class EnsembleEstimator:
    def __init__(self, problem_config, model_configs, weights=None):
        self.problem_config = problem_config
        self.model_config = model_configs
        self.estimator = [
            Estimator(problem_config, config)
            for config in model_configs
        ]
        self.weights = weights if weights is not None else [1] * len(self.estimator)
        self.scores = []

    def train(self, train_data):
        self.scores = []
        for estimator in self.estimator:
            predictor = estimator.train(train_data)
            estimator._predictor = predictor
            if hasattr(estimator, 'evaluate'):
                score = estimator.evaluate(train_data)
                self.scores.append(score)
        return self

    def predict(self, test_data):
        all_preds = []
        for estimator in self.estimator:
            pred = estimator.predict(test_data)
            if hasattr(pred, 'values'):
                pred = pred.values
            elif hasattr(pred, 'samples'):
                pred = pred.samples
            all_preds.append(pred)

        stacked_preds = np.stack(all_preds)
        weights = np.array(self.weights) / np.sum(self.weights)

        return np.average(stacked_preds, axis=0, weights=weights)

    def save(self, path):
        for i, estimator in enumerate(self.estimator):
            estimator.save(f"{path}_estimator_{i}")

    def load(self, path):
        for i, estimator in enumerate(self.estimator):
            estimator.load(f"{path}_estimator_{i}")
