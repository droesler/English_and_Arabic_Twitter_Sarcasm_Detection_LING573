
# CLI Tool
To use the CLI tool run the script with 
```
python src/model_runner.py {train|score} input_file [output_file]
```
This project uses the Pytorch-Lightning framework for running models, which handles significant boilerplate code. There
are many command line options, so for convenience the program allows specifying a config file with all of the options. To 
specify a config file use `-c` or `--config` followed by the path to a JSON file. The corresponding option in a config file will be the
same as the command line flags, but with the initial `--` removed and all hyphens `-` replaced with underscores `_`. The config
file will take priority over any defaults for these options, and any supplied CLI flags will take priority over the config file.

Pytorch-lightning handles backpropagation, learn-rate scheduling, `eval()` and `torch.no_grad()`. The framework also handles putting
models and data on the correct devices and machines, so no need to call `torch.to_device()`. The framework also handles all of the other common
boilerplate code needed when running Pytorch models. The only thing that the user needs to supply is the logic for training and
validation steps, and the initial creation of Dataloaders in the appropriate methods (`train_dataloader`, `val_dataloader`, `predict_dataloader`)

Most of the command line options are passed directly to pytorch_lightning.Trainer and control how the model is run
For more info on Pytorch-Lightning see https://pytorch-lightning.readthedocs.io/en/stable/
## Checkpointing and Logging
The framework will automatically handle tcheckpointing and loggging for training. The location of the logs and checkpoints
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

## Quickstart
Try some of the following command from the root directory:
```
python src/model_runner.py train data/balanced_train_En.csv --log-dir .logging/ -c runner_config.json
```

```
python src/model_runner.py score data/balanced_validation_En.csv --log-dir .logging/ --experiment-name default --experiment-version 1 -c runner_config.json
```

Note that train/validation split will be performed automatically if --val-file is not provided.
```
python src/model_runner.py train data/balanced_train_En.csv --val-file data/balanced_validation_En.csv --accelerator gpu --devices 2 --log-dir .logging/ -c runner_config.json
```