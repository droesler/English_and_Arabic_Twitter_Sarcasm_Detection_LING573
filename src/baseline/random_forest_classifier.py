"""Random Forest Intent Classifier"""
from typing import Tuple, Optional

import logging
import pandas as pd

from scipy.sparse.csr import csr_matrix
from sklearn.ensemble import RandomForestClassifier

from base_classifier import SarcasmClassifier  # pytype: disable=import-error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestIntentClassifier(SarcasmClassifier):
    """Random Forest intent classifier"""

    def __init__(self,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 ngram_range: Tuple = (2, 6),
                 stop_words: Optional[list] = None,
                 n_estimators: int = 1000,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = "auto",
                 max_depth: int = 50,
                 bootstrap: bool = False,
                 random_state: int = 0) -> None:
        """Initialize a random forest intent classifier"""
        super().__init__(train_data, test_data, ngram_range, stop_words, random_state)
        self.algo_name = "RandomForest"
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                         max_depth=self.max_depth, bootstrap=self.bootstrap,
                                         random_state=self.random_state)

    def train(self):
        """Train random forest."""
        self.rf.fit(self.X_train_count, self.y_train)

    def predict(self, predict_data):
        """Predict RandomForest results."""
        if type(predict_data) == csr_matrix:
            return self.rf.predict(predict_data)
        else:
            return self.rf.predict(self.count_vectorizer.transform(predict_data))

    def predict_proba(self, predict_data):
        """ Get predicted probabilities """
        if type(predict_data) == csr_matrix:
            return self.rf.predict_proba(predict_data)
        else:
            return self.rf.predict_proba(self.count_vectorizer.transform(predict_data))
