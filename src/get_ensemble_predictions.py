"""
Performs ensemble model prediction using the logits outputs of our Pytorch Lightning CLI.

The script outputs the following files:
outputs/D3/bert_tweet/model_output.txt - list of predictions for each validation example
results/D3/bert_tweet/model_results.txt - sklearn classification report containing F1 scores
"""

import sys
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


def get_predictions(model_filepaths: list, labels: pd.Series) -> np.array:
  """ 
  Gets voted ensemble predictions.
  """ 

  # get outputs from each child model
  outputs = []
  for filepath in model_filepaths:
    outputs.append(pd.read_csv(filepath))

  # call function to get predictions for each child model
  model_preds = []
  for output in outputs:
    model_preds.append(get_preds_from_logits(output, labels))

  # sum all predictions
  total_preds = np.sum(model_preds, axis=0)

  # get voting results where 3 or more votes is deciding
  voted_preds = np.where(total_preds >= 3, 1, 0)

  return voted_preds


def get_preds_from_logits(output: pd.DataFrame, labels: pd.Series) -> np.array:
  """ 
  Gets predictions from each child model of the ensemble. 
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

  return model_preds


if __name__ == '__main__':

  # set filepaths

  validation_filepath = "data/balanced_validation_En.csv"
  test_filepath = "data/task_A_En_test_column_renamed.csv"

  model1_val_out_path = "outputs/D4/primary/devtest/sub_models/model1_pred_output.csv"
  model2_val_out_path = "outputs/D4/primary/devtest/sub_models/model2_pred_output.csv"
  model3_val_out_path = "outputs/D4/primary/devtest/sub_models/model3_pred_output.csv"
  model4_val_out_path = "outputs/D4/primary/devtest/sub_models/model4_pred_output.csv"
  model5_val_out_path = "outputs/D4/primary/devtest/sub_models/model5_pred_output.csv"

  model1_test_out_path = "outputs/D4/primary/evaltest/sub_models/model1_pred_output.csv"
  model2_test_out_path = "outputs/D4/primary/evaltest/sub_models/model2_pred_output.csv"
  model3_test_out_path = "outputs/D4/primary/evaltest/sub_models/model3_pred_output.csv"
  model4_test_out_path = "outputs/D4/primary/evaltest/sub_models/model4_pred_output.csv"
  model5_test_out_path = "outputs/D4/primary/evaltest/sub_models/model5_pred_output.csv"

  val_outputs_path = "outputs/D4/primary/devtest/model_output.txt"
  val_results_path = "results/D4/primary/devtest/model_results.txt"

  test_outputs_path = "outputs/D4/primary/evaltest/model_output.txt"
  test_results_path = "results/D4/primary/evaltest/model_results.txt"

  val_out_filepaths = [model1_val_out_path, model2_val_out_path, model3_val_out_path, model4_val_out_path, model5_val_out_path]
  test_out_filepaths = [model1_test_out_path, model2_test_out_path, model3_test_out_path, model4_test_out_path, model5_test_out_path]

  # read the validation and test CSV files

  validation_data = pd.read_csv(validation_filepath)
  test_data = pd.read_csv(test_filepath)

  # get ground truth labels

  y_val = validation_data['sarcastic']
  y_test = test_data['sarcastic']

  # call prediction function to get ensemble voted predictions

  val_voted_preds = get_predictions(val_out_filepaths, y_val)
  test_voted_preds = get_predictions(test_out_filepaths, y_test)

  # write outputs and results to files

  with open(val_outputs_path, mode="w", newline="\n", encoding="utf-8") as output_file:
      output_file.write(np.array2string(val_voted_preds, separator=','))

  with open(val_results_path, mode="w", newline="\n", encoding="utf-8") as results_file:
      results_file.write(classification_report(y_val, val_voted_preds))

  with open(test_outputs_path, mode="w", newline="\n", encoding="utf-8") as output_file:
      output_file.write(np.array2string(test_voted_preds, separator=','))

  with open(test_results_path, mode="w", newline="\n", encoding="utf-8") as results_file:
      results_file.write(classification_report(y_test, test_voted_preds))


