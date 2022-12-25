#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ling573-2022-spring

# run weight conversions and primary task predictions
LC_ALL=en_US.UTF-8 python src/model_runner.py convert /home2/droesl/573/2020random_bertweet_large_model.pth  --log-dir .logging --experiment-name model2 --experiment-version 0 -c bertweet_runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_En.csv outputs/D4/primary/devtest/sub_models/model2_pred_output.csv results/D4/primary/devtest/sub_models/model2_metrics.txt  --log-dir .logging/ --experiment-name model2 --experiment-version 0 -c bertweet_runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/task_A_En_test_column_renamed.csv outputs/D4/primary/evaltest/sub_models/model2_pred_output.csv results/D4/primary/evaltest/sub_models/model2_metrics.txt --log-dir .logging/ --experiment-name model2 --experiment-version 0 -c bertweet_runner_config.json

# run the primary task ensemble prediction script
python src/get_thresholded_predictions.py

# run adaptation task predictions
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/balanced_validation_Ar.csv  outputs/D4/adaptation/devtest/ar_pred_output.csv  results/D4/adaptation/devtest/ar_metrics.txt --model-class CamelbertMixClassifier  --log-dir .logging --experiment-name diazhang --experiment-version 1 -c runner_config.json
LC_ALL=en_US.UTF-8 python src/model_runner.py test data/task_A_Ar_test_column_renamed.csv  outputs/D4/adaptation/evaltest/ar_pred_output.csv  results/D4/adaptation/evaltest/ar_metrics.txt --model-class CamelbertMixClassifier  --log-dir .logging --experiment-name diazhang --experiment-version 1 -c runner_config.json
