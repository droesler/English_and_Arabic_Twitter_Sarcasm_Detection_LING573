#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ling573-2022-spring
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/bertweet_large_model.pth --log-dir .logging --experiment-name default --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D3/bert_tweet/pred_output.csv results/D3/bert_tweet/metrics.txt --log-dir .logging/ --experiment-name default --experiment-version 0