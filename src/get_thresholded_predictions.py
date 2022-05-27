"""
Performs thresholded model prediction using the logits outputs of our Pytorch Lightning CLI.

"""

import sys
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


def get_predictions(model_filepath: str, model_result_filepath: str, labels: pd.Series) -> np.array:
  """ 
  Gets thresholded predictions.
  """ 

  # get outputs from each child model
  outputs = pd.read_csv(filepath)

  # call function to get predictions for each child model
  model_preds = get_preds_from_logits(outputs, model_result_filepath, labels)

  return model_preds


def get_preds_from_logits(output: pd.DataFrame, results_paths: str, labels: pd.Series) -> np.array:
  """ 
  Gets raw predictions from model, calculates optimal threshold, and returns thresholded predictions. 
  Called by get_predictions.
  """ 
  logits0 = output['logits_0']
  logits1 = output['logits_1']

  model_logits = []
  for first, second in zip(logits0, logits1):
    model_logits.append([first, second])
  
  model_logits_tensor = torch.Tensor(model_logits)
  model_probs = F.softmax(model_logits_tensor, dim=1).cpu().numpy()

  # get positive probs only to calculate best threshold for positive f1
  pos_probs = model_probs[:, 1]
  precision, recall, thresholds = precision_recall_curve(labels, pos_probs)

  # calculate f1 for the positive class, ignore divide by zero warnings
  with np.errstate(invalid='ignore'):
    pos_f1_score = (2 * precision * recall) / (precision + recall)

  # retrieve the best threshold for the best positive f1 score
  ix = np.argmax(np.nan_to_num(pos_f1_score))
  best_threshold = thresholds[ix]

  model_preds = np.where(model_probs[:, 1] >= best_threshold, 1, 0)

  # update sub-model results file with thresholded version of predictions
  with open(results_paths, mode="w", newline="\n", encoding="utf-8") as results_file:
      results_file.write(classification_report(labels, model_preds))

  return model_preds


if __name__ == '__main__':

  # set filepaths

  validation_filepath = "data/balanced_validation_En.csv"
  test_filepath = "data/task_A_En_test_column_renamed.csv"

  model2_val_out_path = "outputs/D4/primary/devtest/sub_models/model2_pred_output.csv"
  model2_test_out_path = "outputs/D4/primary/evaltest/sub_models/model2_pred_output.csv"
  model2_val_result_path = "results/D4/primary/devtest/sub_models/model2_metrics.txt"
  model2_test_result_path = "results/D4/primary/evaltest/sub_models/model2__metrics.txt"

  val_outputs_path = "outputs/D4/primary/devtest/model_output.txt"
  val_results_path = "results/D4/primary/devtest/model_results.txt"

  test_outputs_path = "outputs/D4/primary/evaltest/model_output.txt"
  test_results_path = "results/D4/primary/evaltest/model_results.txt"

  # read the validation and test CSV files

  validation_data = pd.read_csv(validation_filepath)
  test_data = pd.read_csv(test_filepath)

  # get ground truth labels

  y_val = validation_data['sarcastic']
  y_test = test_data['sarcastic']

  # call prediction function to get thresholded predictions

  val_thresholded_preds = get_predictions(model2_val_out_path, model2_val_result_path, y_val)
  test_thresholded_preds = get_predictions(model2_test_out_path, model2_test_result_path, y_test)

  # write outputs and results to files

  with open(val_outputs_path, mode="w", newline="\n", encoding="utf-8") as output_file:
      output_file.write(np.array2string(val_thresholded_preds, separator=','))

  with open(val_results_path, mode="w", newline="\n", encoding="utf-8") as results_file:
      results_file.write(classification_report(y_val, val_thresholded_preds))

  with open(test_outputs_path, mode="w", newline="\n", encoding="utf-8") as output_file:
      output_file.write(np.array2string(test_thresholded_preds, separator=','))

  with open(test_results_path, mode="w", newline="\n", encoding="utf-8") as results_file:
      results_file.write(classification_report(y_test, test_thresholded_preds))


