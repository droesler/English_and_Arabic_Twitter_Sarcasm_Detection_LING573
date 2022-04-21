""" LightGBM Intent Classifier """
from typing import Tuple, Optional

import logging
import numpy as np
import pandas as pd

import lightgbm as lgb
from scipy.sparse.csr import csr_matrix

from base_classifier import SarcasmClassifier  # pytype: disable=import-error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMClassifier(SarcasmClassifier):
    """LightGBM intent classifier"""

    def __init__(self,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 ngram_range: Tuple = (2, 6),
                 stop_words: Optional[list] = None,
                 random_state: int = 0,
                 num_leaves: int = 500):
        """Initialize a lightGBM intent classifier"""
        super().__init__(train_data, test_data, ngram_range, stop_words, random_state)
        self.algo_name = "lightGBM"
        self.num_leaves = num_leaves
        self.lgbm = lgb.LGBMClassifier(
            random_state=self.random_state,
            num_leaves=self.num_leaves,
            class_weight='balanced',
            is_unbalance=True
        )

    def train(self):
        """Train lightGBM."""
        X_train_count_lgbm = self.X_train_count.astype('float32')
        self.lgbm.fit(X_train_count_lgbm, self.y_train)

    def predict(self, predict_data):
        """Predict lightGBM results."""
        if type(predict_data) == csr_matrix:
            return self.lgbm.predict(predict_data)
        else:
            return self.lgbm.predict(self.count_vectorizer.transform(predict_data).astype('float32'))

    def predict_proba(self, predict_data):
        """ Get predicted probabilities """
        if type(predict_data) == csr_matrix:
            return self.lgbm.predict_proba(predict_data)
        else:
            return self.lgbm.predict_proba(self.count_vectorizer.transform(predict_data).astype('float32'))

    def predict_with_argmax_filler(self, predict_data):
        """ Predict and fill with argmax result if prediction is empty """
        y_prob = self.predict_proba(predict_data)
        y_pred = self.predict(predict_data)
        for i in range(len(predict_data)):
            if all(y_pred[i] == 0):
                max_idx = np.argmax(y_prob[i])
                y_pred[i][max_idx] = 1
        return y_pred
