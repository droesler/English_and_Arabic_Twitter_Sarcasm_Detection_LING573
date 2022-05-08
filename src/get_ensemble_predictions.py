"""
Performs ensemble model prediction using the logits outputs of our Pytorch Lightning CLI.
The command line for launching the script is:
python get_ensemble_preictions.py <validation set file path> <model1 output filepath> <model2 output filepath> <model3 output filepath> <model4 output filepath> <model5 output filepath>

After git cloning the repository these are:

<validation set file path>: data/balanced_validation_En.csv
<model1 output filepath> outputs/D3/bert_tweet/sub_models/model1_pred_output.csv
<model2 output filepath> outputs/D3/bert_tweet/sub_models/model2_pred_output.csv
<model3 output filepath> outputs/D3/bert_tweet/sub_models/model3_pred_output.csv
<model4 output filepath> outputs/D3/bert_tweet/sub_models/model4_pred_output.csv
<model5 output filepath> outputs/D3/bert_tweet/sub_models/model5_pred_output.csv

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


if __name__ == '__main__':

  model_threshold_list = [0.09233454, 0.45811993 , 0.33289504  , 0.29287043, 0.49070767]

  validation_filepath = sys.argv[1]
  model_filepaths = [sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]

  # read the validation CSV file
  validation_data = pd.read_csv(validation_filepath)
  
  # get ground truth validation labels
  y_val = validation_data['sarcastic']

  # get outputs from each child model
  outputs = []
  for filepath in model_filepaths:
    outputs.append(pd.read_csv(filepath))

  # define function for getting predictions from logits
  def get_preds_from_logits(output: pd.DataFrame, threshold: float) -> np.array:
    
    logits0 = output['logits_0']
    logits1 = output['logits_1']

    model_logits = []
    for first, second in zip(logits0, logits1):
      model_logits.append([first, second])
    
    model_logits_tensor = torch.Tensor(model_logits)
    model_probs = F.softmax(model_logits_tensor, dim=1).cpu().numpy()
    model_preds = np.where(model_probs[:, 1] >= threshold, 1, 0)

    return model_preds

  # call function for each child model
  model_preds = []
  for output, threshold in zip(outputs, model_threshold_list):
    model_preds.append(get_preds_from_logits(output, threshold))

  # sum all predictions
  total_preds = np.sum(model_preds, axis=0)

  # get voting results where 3 or more votes is deciding
  voted_preds = np.where(total_preds >= 3, 1, 0)

  with open("outputs/D3/bert_tweet/model_output.txt", mode="w", newline="\n", encoding="utf-8") as output_file:
      output_file.write(np.array2string(voted_preds, separator=','))

  with open("results/D3/bert_tweet/model_results.txt", mode="w", newline="\n", encoding="utf-8") as results_file:
      results_file.write(classification_report(y_val, voted_preds))

