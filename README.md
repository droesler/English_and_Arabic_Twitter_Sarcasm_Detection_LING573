# ling573-2022-spring


## Team

- Amy Tzu-Yu Chen @amy17519
- David Roesler @droesler
- Diana Baumgartner-Zhang @Diana-BZ
- Juliana McCausland @ju-mc

## How to Replicate D3 Results

1. First, clone the project repo at https://github.com/amy17519/ling573-2022-spring.git

```
git clone https://github.com/amy17519/ling573-2022-spring.git
```

2. We use conda to manage our dev environment. To rerun our evaluation script, you can run the following command to set up the environment first:

```
conda env create -f src/environment.yml
```

3. Make sure you can execute `D3.sh`

```
chmod +x D3.sh
```

4. After the conda environment is activated and you have the right permission for `D3.sh`, you can run the `D3.sh` file to replicate all the steps for both baseline models and BertTweet model.

```
bash D3.sh
```

or via condor

```
condor_submit D3.cmd
```


## What's in the `D3.sh`

### Replicate Results for D3 on patas - BertTweet-large Model

The BertTweet-large model is saved at `/home2/droesl/573/bertweet_large_model.pth ` on patas and we already updated the permission for the model to be accessible by graders. You will have to convert the model to be PyTorch Lightning checkpoint before running evaluation step.

```
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/bertweet_large_model.pth --log-dir .logging --experiment-name default --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D3/bert_tweet/pred_output.csv results/D3/bert_tweet/metrics.txt --log-dir .logging/ --experiment-name default --experiment-version 0
```

The following are the outputs from the execution:

- `results/D3/bert_tweet/metrics.txt` - the classification report for BERTweet model
- `outputs/D3/bert_tweet/pred_output.csv` - contains scores from running BERTweet model
