""" Base - Intent Classifier """
from typing import Tuple, Optional

import logging
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SarcasmClassifier():
    """Base class - sarcasm classifier"""

    def __init__(self,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 ngram_range: Tuple = (2, 6),
                 stop_words: Optional[list] = None,
                 random_state: int = 0):
        """Initialize an intent classifier"""
        self.train_data = train_data
        self.test_data = test_data
        self.random_state = random_state
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_data['tweet'], self.train_data['sarcastic'], self.test_data['tweet'], self.test_data['sarcastic']
        self.count_vectorizer = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=10000
        )
        self.X_train_count = self.fit_count_vectorizer()

    def fit_count_vectorizer(self):
        """ Fit count vectorizer """
        return self.count_vectorizer.fit_transform(self.X_train)

    def get_classification_report(self, y_true, y_pred):
        """ Get classification report """
        report = classification_report(y_true, y_pred, zero_division=0)
        logger.info(report)
        return report
