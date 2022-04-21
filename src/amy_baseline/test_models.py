""" Run multiple classifiers. `python intent_classifier_pipeline.py 2>&1 | tee ./model.log` to see and save logs"""
import logging
import pickle

# pytype: disable=import-error
import pandas as pd
from lightgbm_classifier import LightGBMClassifier
from random_forest_classifier import RandomForestIntentClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    df_train = pd.read_csv("../../data/balanced_train_En.csv")
    df_test = pd.read_csv("../../data/balanced_validation_En.csv")
    for random_state in [2]:
        logger.info(f"Started training rf with random_state = {random_state}")

        rf = RandomForestIntentClassifier(train_data=df_train, test_data=df_test, random_state=random_state)
        rf.train()
        y_pred_rf = rf.predict(rf.X_test)
        rf.get_classification_report(rf.y_test, y_pred_rf)

        # rf_fname = f'./pickle/rf_random_{random_state}.pkl'
        # with open(rf_fname, 'wb') as p:
        #     pickle.dump(rf, p)

        logger.info(f"Started training lightgbm with and random_state = {random_state}")

        lg = LightGBMClassifier(train_data=df_train, test_data=df_test, random_state=random_state)
        lg.train()
        y_pred_lg = lg.predict(lg.X_test)
        lg.get_classification_report(lg.y_test, y_pred_lg)

        # lg_fname = f'./pickle/lg_random_{random_state}.pkl'
        # with open(lg_fname, 'wb') as p:
        #     pickle.dump(lg, p)
