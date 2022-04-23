""" Run multiple classifiers. `python intent_classifier_pipeline.py 2>&1 | tee ./model.log` to see and save logs"""
import logging
import pickle

# pytype: disable=import-error
import pandas as pd
from lightgbm_classifier import LightGBMClassifier
from random_forest_classifier import RandomForestIntentClassifier
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    df_train = pd.read_csv("../../data/balanced_train_En.csv")
    df_test = pd.read_csv("../../data/balanced_validation_En.csv")
    random_state = 2

    rf = RandomForestIntentClassifier(train_data=df_train, test_data=df_test, random_state=random_state)
    rf.train()
    y_pred_rf = rf.predict(rf.X_test)
    y_pred_rf_scores = rf.predict_proba(rf.X_test)
    rf_report = rf.get_classification_report(rf.y_test, y_pred_rf)
    rf_metrics = {
        'test_Accuracy': rf_report["accuracy"],
        'test_Precision': rf_report["macro avg"]["precision"],
        'test_Recall': rf_report["macro avg"]["recall"],
        'test_F1Score': rf_report["macro avg"]["f1-score"],
        'test_AUROC': roc_auc_score(rf.y_test, y_pred_rf_scores[:,1])
    }
    df_rf = pd.DataFrame(
        data= {
            "preds": y_pred_rf,
            "labels": rf.y_test.tolist(),
            "logits_0": [scores[0] for scores in y_pred_rf_scores],
            "logits_1": [scores[1] for scores in y_pred_rf_scores]
        }
    )
    df_rf.to_csv("../../results/D2/baseline_rf/pred_output.csv", index=False)
    pd.DataFrame.from_dict(rf_metrics, orient='index', columns = ["Value"]).to_csv("../../results/D2/baseline_rf/metrics_output.csv", index_label="Metric Name")
    rf_fname = f'../../results/D2/baseline_rf/rf.pkl'
    with open(rf_fname, 'wb') as p:
        pickle.dump(rf, p)


    lg = LightGBMClassifier(train_data=df_train, test_data=df_test, random_state=random_state)
    lg.train()
    y_pred_lg = lg.predict(lg.X_test)
    y_pred_lg_scores = lg.predict_proba(lg.X_test)
    lg_report = lg.get_classification_report(lg.y_test, y_pred_lg)
    lg_metrics = {
        'test_Accuracy': lg_report["accuracy"],
        'test_Precision': lg_report["macro avg"]["precision"],
        'test_Recall': lg_report["macro avg"]["recall"],
        'test_F1Score': lg_report["macro avg"]["f1-score"],
        'test_AUROC': roc_auc_score(lg.y_test, y_pred_lg_scores[:,1])
    }
    df_lg = pd.DataFrame(
        data= {
            "preds": y_pred_lg,
            "labels": lg.y_test.tolist(),
            "logits_0": [scores[0] for scores in y_pred_lg_scores],
            "logits_1": [scores[1] for scores in y_pred_lg_scores]
        }
    )
    df_lg.to_csv("../../results/D2/baseline_lightgbm/pred_output.csv", index=False)
    pd.DataFrame.from_dict(lg_metrics, orient='index', columns = ["Value"]).to_csv("../../results/D2/baseline_lightgbm/metrics_output.csv", index_label="Metric Name")
    lg_fname = f'../../results/D2/baseline_lightgbm/lg.pkl'
    with open(lg_fname, 'wb') as p:
        pickle.dump(lg, p)