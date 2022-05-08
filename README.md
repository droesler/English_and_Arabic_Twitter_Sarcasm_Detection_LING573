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

### Replicate Results for D3 on patas - Ensemble of 5 BertTweet-large models

The five BertTweet-large models are saved in the following locations on Patas:
`/home2/droesl/573/1010random_bertweet_large_model.pth` 
`/home2/droesl/573/2020random_bertweet_large_model.pth`
`/home2/droesl/573/4040random_bertweet_large_model.pth`
`/home2/droesl/573/6060random_bertweet_large_model.pth`
`/home2/droesl/573/7070random_bertweet_large_model.pth`
We have set the permissions for the saved models to be accessible to graders. The D3.sh script will convert each of the five models to PyTorch Lightning checkpoints before performing predictions.

```
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/1010random_bertweet_large_model.pth  --log-dir .logging --experiment-name model1 --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D3/bert_tweet/sub_models/model1_pred_output.csv results/D3/bert_tweet/sub_models/model1_metrics.txt --log-dir .logging/ --experiment-name model1 --experiment-version 0

LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/2020random_bertweet_large_model.pth  --log-dir .logging --experiment-name model2 --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D3/bert_tweet/sub_models/model2_pred_output.csv results/D3/bert_tweet/sub_models/model2_metrics.txt  --log-dir .logging/ --experiment-name model2 --experiment-version 0

LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/4040random_bertweet_large_model.pth  --log-dir .logging --experiment-name model3 --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D3/bert_tweet/sub_models/model3_pred_output.csv results/D3/bert_tweet/sub_models/model3_metrics.txt  --log-dir .logging/ --experiment-name model3 --experiment-version 0

LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/6060random_bertweet_large_model.pth  --log-dir .logging --experiment-name model4 --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D3/bert_tweet/sub_models/model4_pred_output.csv results/D3/bert_tweet/sub_models/model4_metrics.txt  --log-dir .logging/ --experiment-name model4 --experiment-version 0

LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/7070random_bertweet_large_model.pth  --log-dir .logging --experiment-name model5 --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D3/bert_tweet/sub_models/model5_pred_output.csv results/D3/bert_tweet/sub_models/model5_metrics.txt  --log-dir .logging/ --experiment-name model5 --experiment-version 0

```
After output logits are produced by the five models, another script 'src/get_ensemble.predictions.py' will collect the output logits from the five models and do majority voting to determine the final ensemble predictions.


The following files show the predictions and scores of the entire ensemble:

- `results/D3/bert_tweet/metrics.txt` - the classification report for the ensemble
- `outputs/D3/bert_tweet/pred_output.csv` - the majority vote predictions of the ensemble

For each of the five models in the ensemble, sub-predictions and scores can be found in 

- `results/D3/bert_tweet/sub_models/`
- `outputs/D3/bert_tweet/sub_models/`

The notebooks used to train each of the models on colab can also be found in:

- `src/training_notebooks/D3_training_notebooks`
