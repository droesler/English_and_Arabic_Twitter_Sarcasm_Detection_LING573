
# Conda Environment Setup
To install the necessary packages to run the cli tool, an environment file has been provided.
From the root project directory (ling573-2022-spring/) run:
```
conda env create -f src/environment.yml
```
This will create a conda environment with the name "ling573-2022-spring". To activate the environment, run: 
```
conda activate ling573-2022-spring
```
If you update `environment.yml`, run:
```
conda activate ling573-2022-spring
conda env update --file src/environment.yml --prune
```


# Quick Start
Train a model on *balanced_train_En.csv* using options from runner_config.json, logging to ./logging, and using
BertSmallClassifier as the model. Since no --val-file is provided, balanced_train_En.csv will be split into train and
validation using `--train-size` option as the percent used for training.
```
python src/model_runner.py train data/balanced_train_En.csv --log-dir .logging/ -c runner_config.json
```

Train a model on balanced_train_En.csv using options from runner_config.json, logging to ./logging, and using
the large BertClassifier as the model.
```
python src/model_runner.py train data/balanced_train_En.csv --model-class BertClassifier --log-dir .logging/ -c runner_config.json
```

Run testing on *balanced_validation_En.csv* using a previously saved model in *.logging/default/version_1* and save the predictions to *pred_output.csv*
and metrics to *metrics_output.csv*.
```
python src/model_runner.py test data/balanced_validation_En.csv results/pred_output.csv results/metric_output.csv --log-dir .logging/ --experiment-name default --experiment-version 1
```

Score the dataset balanced_validation_En.csv using the model saved at .logging/default/version_1 with options loaded from
runner_config.json
```
python src/model_runner.py predict data/balanced_validation_En.csv --log-dir .logging/ --experiment-name default --experiment-version 1 -c runner_config.json
```
Train a model using balanced_train_En.csv as training data and balanced_validation_En.csv as the validation data. 2 gpus
will be used for scoring data.
```
python src/model_runner.py train data/balanced_train_En.csv --val-file data/balanced_validation_En.csv --accelerator gpu --devices 2 --log-dir .logging/ -c runner_config.json
```
Train a model using balanced_train_En.csv as training data and balanced_validation_En.csv as the validation data. 2 gpus
will be used for scoring data.
```
python src/model_runner.py train data/balanced_train_En.csv --val-file data/balanced_validation_En.csv --accelerator gpu --devices 2 --log-dir .logging/ -c runner_config.json
```

# CLI Tool

To use the CLI tool, cd to the root project directory:
```
cd path/to/proj/ling573-2022-spring
```
The CLI script can be run with the following format:
```
python src/model_runner.py {train|test|predict|convert} input_file [output_pred_file] [output_metrics_file] [optional flags]
```
This project uses the Pytorch-Lightning framework for running models, which handles significant boilerplate code. There
are many command line options, so for convenience the program allows specifying a config file with all of the options. To 
specify a config file use `-c` or `--config` followed by the path to a JSON file. The corresponding option in a config file will be the
same as the command line flags, but with the initial `--` removed and all hyphens `-` replaced with underscores `_`. The config
file will take priority over any defaults for these options, and any supplied CLI flags will take priority over the config file.

Pytorch-lightning handles back-propagation, learn-rate scheduling, `eval()` and `torch.no_grad()`. The framework also handles putting
models and data on the correct devices and machines, so no need to call `torch.to_device()`. The framework also handles all of the other common
boilerplate code needed when running Pytorch models. The only thing that the user needs to supply is the logic for training and
validation steps, and the initial creation of Dataloaders in the appropriate methods (`train_dataloader`, `val_dataloader`, `predict_dataloader`)

Most of the command line options are passed directly to pytorch_lightning.Trainer and control how the model is run
For more info on Pytorch-Lightning see https://pytorch-lightning.readthedocs.io/en/stable/

## Checkpointing and Logging
The framework will automatically handle checkpointing and logging for training. The location of the logs and checkpoints
specified with the options `--log-dir`, `--experiment-name`, `--experiment-version`. A directory hierarchy 
will be created with these three components like so: `my_log_dir/my_experiment_name/version_1`. The version argument should
be an integer. If `--experiment-version` is not provided, the framework will automatically increment the highest version by 1 so
that previous models are left unchanged. If all three arguments are supplied and checkpoint and logs already exist in that location,
then that checkpoint will be loaded and training will resume where it left off. This gives modeling error tolerance. For scoring, 
these three parameters are required in order to locate a checkpointed model to load to perform scoring with. Models will be saved
only when `avg_val_loss` improves from the last time the model was checkpointed. This way we are not saving an overfit model.

Logs will automatically keep track of the hyperparameters that are passed to the command line and will keep track of metrics and losses
for training/validation steps and epochs. These logs and charts of the metrics can be viewed by running Tensorboard like so:
```
tensorboard --logdir=ling573-2022-spring/.logging/
```
The `--logdir` directory should be the same as the one provided to the `--log-dir` option of model_runner.py.

To connect to the remote tensorboard on dryas:

```
ssh -L 6006:localhost:6006 yourusername@dryas.ling.washington.edu
```
Once you have ssh-ed, type into your browser bar:

```
localhost:6006
```

## Testing and Inference
The `test` and `predict` are used for running the model outside of the training/validation loop. These modes will put
the model into the `eval()` state and turn off tracking of gradients for efficient scoring. For both of these modes you 
will need to provide the options for `--log-dir`, `--experiment-name`, and `--experiment-version` so that the model can
be loaded for scoring. These two modes behave similarly, but `test` is used when you have the labels and you want to evaluate
the performance of a saved model. `predict` is used when you want to perform inference and do not have the labels.

Test mode can be run like this:
```
python src/model_runner.py test test_input_file pred_output_file metric_output_file --log-dir .logging --experiment-name prod --experiment-version 1
```

Predict mode can be run like this:
```
python src/model_runner.py predict input_file pred_output_file --log-dir .logging --experiment-name prod --experiment-version 1
```

## Convert
If a model is not saved using the Pytorch-Lightning framework, there is a command to convert an existing model to a checkpoint.
Any command line options that are passed will be saved as the model hyperparameters. However, since we are converting an already
existing model, these hparams may not reflect the conditions under which the original model was trained. The usual `--log-dir`
`--experiment-name`, `--experiment-version` options need to be passed to specify where the checkpoint should be saved.

To convert a saved model at model/to/convert.pth to a Pytorch-Lightning checkpoint at .logging/default/version_0/, run the following command:
```
python src/model_runner.py convert model/to/convert.pth --log-dir .logging --experiment-name default --experiment-version 0 -c runner_config.json
```


# Changing Models
When adding new models to model runner, make sure to do the following:
1) Add the new model class to models.py
2) Add a static method to the class called `get_preprocessor` that returns a function to preprocess the input data.
3) import the new model class in model_runner.py
4) If this model differs from other the other Bert models, you may need to change the logic in `train_step`, `val_step`, `predict_step`, and `forward`. In most cases this won't be necessary.

# Notes
05/02) Addition of train.cmd, and train.sh

In train.sh, logging goes into shared folder on patas; some additional steps need to be added to access the tensorboard (port 6006)

For train.cmd, requests gpu. Occasionally runs into error where it claims "no gpu available" although it has been assigned a gpu. May be an issue with a specific machine.

05/13) models.py, addition of MarbertClassifier and CamelbertMixClassifier for Adaptation task

runner_config.json, addition of changeable model_params (e.g. dropout, freeze_bert, hidden_layers)
