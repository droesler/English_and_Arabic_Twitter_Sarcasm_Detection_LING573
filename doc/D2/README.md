## How Set up Environment

First, clone the project repo at https://github.com/amy17519/ling573-2022-spring.git

```
git clone https://github.com/amy17519/ling573-2022-spring.git
```

We use conda to manage our dev environment. To rerun our evaluation script, you can run the following command to set up the environment first:

```
conda env create -f src/environment.yml
conda activate ling573-2022-spring
```

Make sure you can execute `D2.sh`

```
chmod +x D2.sh
```

After the conda environment is activated and you have the right permission for `D2.sh`, you can run the `D2.sh` file to replicate all the steps for both baseline models and BertTweet model.

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
# Baseline Models
python src/baseline/test_models.py
```

### Replicate Results for D2 on patas - BertTweet Models

```
# make sure you are at root of the project
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/test_model.pth --log-dir .logging --experiment-name default --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D2/bert_tweet/pred_output.csv results/D2/bert_tweet/metrics.txt --log-dir .logging/ --experiment-name default --experiment-version 0
```
