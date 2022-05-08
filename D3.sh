#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ling573-2022-spring

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

# run the ensemble prediction script
python src/get_ensemble_predictions.py data/balanced_validation_En.csv outputs/D3/bert_tweet/sub_models/model1_pred_output.csv outputs/D3/bert_tweet/sub_models/model2_pred_output.csv outputs/D3/bert_tweet/sub_models/model3_pred_output.csv outputs/D3/bert_tweet/sub_models/model4_pred_output.csv outputs/D3/bert_tweet/sub_models/model5_pred_output.csv
