# English and Arabic Sarcasm Detection in Tweets
Group project for University of Washington LING573 NLP Systems and Applications Spring 2022

We build English and Arabic Sarcasm Detection systems using the Twitter sarcasm dataset of SemEval 2022 Task 6, iSarcasmEval ([Oprea and Magdy, 2020](https://aclanthology.org/2022.semeval-1.111.pdf)). Our system outperforms the top scoring submission for the English portion of the task.  
- Our [paper](https://github.com/droesler/English_and_Arabic_Twitter_Sarcasm_Detection_LING573/blob/main/doc/English%20and%20Arabic%20Sarcasm%20Detection%20in%20Tweets%20(paper).pdf) and [slides](https://github.com/droesler/English_and_Arabic_Twitter_Sarcasm_Detection_LING573/blob/main/doc/English%20and%20Arabic%20Sarcasm%20Detection%20in%20Tweets%20(slides).pdf) for the project.

## Team

- Amy Tzu-Yu Chen @amy17519
- David Roesler @droesler
- Diana Baumgartner-Zhang @Diana-BZ
- Juliana McCausland @ju-mc

## Instructions for running models on University of Washington's Patas/Dryas Linux cluster:

1. First, clone the project repo at https://github.com/droesler/English_and_Arabic_Twitter_Sarcasm_Detection_LING573.git  

```
git clone https://github.com/droesler/English_and_Arabic_Twitter_Sarcasm_Detection_LING573.git
```

2. Please remove any previously created 'ling573-2022-spring' environment and run the following command to re-build the updated environment:

```
conda env create -f src/environment.yml
```

3. Make sure you can execute `D4.sh`

```
chmod +x D4.sh
```

4. After the conda environment is activated and you have the right permission for `D4.sh`, you can run the `D4.sh` file to perform predictions using our D4 models.
```
bash D4.sh
```

or via condor

```
condor_submit D4.cmd
```


## What's in the `D4.sh`

## Primary task:

Our primary D4 model is a fine-tuned BertTweet-large model. 

The BertTweet-large model is saved in the following location on Patas:

`/home2/droesl/573/2020random_bertweet_large_model.pth`  

We have set the permissions for the saved model to be accessible to graders. 

The D4.sh script will convert the model to PyTorch Lightning checkpoints before performing predictions on both the dev and test sets:
```
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/2020random_bertweet_large_model.pth  --log-dir .logging --experiment-name model2 --experiment-version 0 -c bertweet_runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D4/primary/devtest/sub_models/model2_pred_output.csv results/D4/primary/devtest/sub_models/model2_metrics.txt  --log-dir .logging/ --experiment-name model2 --experiment-version 0 -c bertweet_runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/task_A_En_test_column_renamed.csv outputs/D4/primary/evaltest/sub_models/model2_pred_output.csv results/D4/primary/evaltest/sub_models/model2_metrics.txt --log-dir .logging/ --experiment-name model2 --experiment-version 0 -c bertweet_runner_config.json

```

After output logits are produced by the model, another script, `src/get_thresholded.predictions.py`, will collect the output logits from the model and use thresholding to produce the final predictions
     
D4.sh outputs the following results and output files:

- `results/D4/primary/devtest/model_results.txt` - the classification report for the development set
- `results/D4/primary/evaltest/model_results.txt` - the classification report for the test set
- `outputs/D4/primary/devtest/model_output.txt` - predictions on the development set
- `outputs/D4/primary/devtest/model_output.txt` - predictions on the test set

The notebooks used to fine-tune the BERTweet models on colab can be found in:

- `src/training_notebooks/D3_training_notebooks`

## Adaptation task:

Our Adaptation Task model is a fine-tuned CAMeLBERT-Mix model.

The CAMeLBERT-Mix model is saved in the following location on Patas:
`/home2/diazhang/LING573/logging/`  

The version in D4 is version_1

We have set the permissions for the saved model to be accessible to graders.

The D4.sh script will also perform predictions on both the dev and test sets using our adaptation model and will produce the following results and output files:

- `results/D4/adaptation/devtest/ar_metrics.txt` - the classification report for the adaptation model on the development set
- `results/D4/adaptation/evaltest/ar_metrics.txt` - the classification report for the adaptation model on the test set
- `outputs/D4/adaptation/devtest/ar_pred_output.csv` - the predictions of the adaptation model on the development set
- `outputs/D4/adaptation/devtest/ar_pred_output.csv` - the predictions of the adaptation model on the test set


