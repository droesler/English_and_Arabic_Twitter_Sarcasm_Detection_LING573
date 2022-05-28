# ling573-2022-spring


## Team

- Amy Tzu-Yu Chen @amy17519
- David Roesler @droesler
- Diana Baumgartner-Zhang @Diana-BZ
- Juliana McCausland @ju-mc

## How to Replicate D4 Results

1. First, clone the project repo at https://github.com/amy17519/ling573-2022-spring.git

```
git clone https://github.com/amy17519/ling573-2022-spring.git
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

The D4.sh script will also perform predictions on both the dev and test sets using our adaptation model and will produce the following results and output files:

- `results/D4/adaptation/devtest/model_results.txt` - the classification report for the adaptation model on the development set
- `results/D4/adaptation/evaltest/model_results.txt` - the classification report for the adaptation model on the test set
- `outputs/D4/adaptation/devtest/model_output.txt` - the predictions of the adaptation model on the development set
- `outputs/D4/adaptation/devtest/model_output.txt` - the predictions of the adaptation model on the test set


