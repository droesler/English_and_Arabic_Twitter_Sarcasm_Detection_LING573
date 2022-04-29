#!/bin/sh
# Note that sometimes requesting a gpu on patas results in an error where pytorch cannot see the gpu.
# This may be an issue with a particular machine on patas
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ling573-2022-spring
LC_ALL=en_US.UTF-8 python src/model_runner.py train data/balanced_train_En.csv --accelerator gpu --devices 1 --model-class BertClassifier --log-dir /home2/diazhang/LING573/logging \
--experiment-name $USER -c runner_config.json
