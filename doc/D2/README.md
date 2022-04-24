## How Set up Environment

1. First, clone the project repo at https://github.com/amy17519/ling573-2022-spring.git

```
git clone https://github.com/amy17519/ling573-2022-spring.git
```

2. We use conda to manage our dev environment. To rerun our evaluation script, you can run the following command to set up the environment first:

```
conda env create -f src/environment.yml
```

3. Make sure you can execute `D2.sh`

```
chmod +x D2.sh
```

4. After the conda environment is activated and you have the right permission for `D2.sh`, you can run the `D2.sh` file to replicate all the steps for both baseline models and BertTweet model.

```
bash D2.sh
```

or via condor

```
condor_submit D2.cmd
```


## What's in the `D2.sh`

### Replicate Results for D2 on patas - Baseline Models

We use random forest and lightgbm to train our baseline models. You can view the results in `/results/D2/baseline_rf` and `/results/D2/baseline_lightgbm` or replicate the results by running the following command. Make sure you are under directory `/src/baseline`

```
python src/baseline/test_models.py
```

The following are the outputs from the execution:

- `results/D2/baseline_lightgbm/metrics.txt` - the classification report for baseline lightgbm model
- `results/D2/baseline_rf/metrics.txt` - the classification report for baseline random forest model
- `outputs/D2/baseline_lightgbm/D2_scores.out` - containing scores from running baseline lightgbm model
- `outputs/D2/baseline_rf/D2_scores.out` - containing scores from running baseline random forest model
- `outputs/D2/baseline_rf/rf.pkl` - the pickle file for baseline random forest model
- `outputs/D2/baseline_lightgbm/lg.pkl` - the pickle file for baseline random forest model


### Replicate Results for D2 on patas - BertTweet Models

The BertTweet model is saved at `/home2/droesl/573/test_model.pth` on patas and we already updated the permission for the model to be accessible by graders. You will have to convert the model to be PyTorch Lightning checkpoint before running evaluation step.

```
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/test_model.pth --log-dir .logging --experiment-name default --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D2/bert_tweet/pred_output.csv results/D2/bert_tweet/metrics.txt --log-dir .logging/ --experiment-name default --experiment-version 0
```

The following are the outputs from the execution:

- `results/D2/bert_tweet/metrics.txt` - the classification report for BertTweet model
- `outputs/D2/bert_tweet/D2_scores.out` - containing scores from running baseline lightgbm model