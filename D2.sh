#!/bin/sh
# conda env create -f src/environment.yml
conda init
conda activate ling573-2022-spring
python src/baseline/test_models.py
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/test_model.pth --log-dir .logging --experiment-name default --experiment-version 0 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D2/bert_tweet/pred_output.csv results/D2/bert_tweet/metrics.txt --log-dir .logging/ --experiment-name default --experiment-version 0