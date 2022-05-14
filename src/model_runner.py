import argparse
import os
import copy
import sys
import json
import inspect
import numpy as np
import pandas as pd
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score
from models import BertClassifier, BertSmallClassifier, BertweetLargeClassifier, ArbertClassifier, CamelbertMixClassifier
from sklearn.metrics import classification_report


class DataRetriever:
    def __init__(self, label_to_classify=None, train_file=None, val_file=None, pred_file=None, test_file=None,
                 random_state=None, train_size=0.8):
        if random_state is None or np.isnan(random_state):
            random_state = np.random.randint(0, 1e8)
        self.random_state = int(random_state)
        self.label_to_classify = label_to_classify
        self.__train_mask = None
        self.train_size = train_size
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.pred_file = pred_file

    def get_train(self, delimiter=','):
        if self.val_file is None:
            train_data = self._split_train_val(return_train=True)
        else:
            train_data = pd.read_csv(self.train_file, sep=delimiter)
        train_data = train_data.loc[train_data['tweet'].notna(), :]
        return train_data['tweet'], train_data[self.label_to_classify].astype(int)

    def get_val(self, delimiter=','):
        if self.val_file is None:
            val_data = self._split_train_val(return_train=False)
        else:
            val_data = pd.read_csv(self.val_file, sep=delimiter)
        val_data = val_data.loc[val_data['tweet'].notna(), :]
        return val_data['tweet'], val_data[self.label_to_classify].astype(int)

    def get_test(self, delimiter=','):
        test_data = pd.read_csv(self.test_file, sep=delimiter, dtype=str)
        return test_data['tweet'], test_data[self.label_to_classify].astype(int)

    def get_pred(self, delimiter=','):
        pred_data = pd.read_csv(self.pred_file, sep=delimiter, dtype=str)
        return pred_data['tweet']

    def _split_train_val(self, return_train=True, delimiter=','):
        data = pd.read_csv(self.train_file, sep=delimiter)
        # If the mask is set, then we've already split the data. We need to be consistent with the split so we keep the mask as is.
        if self.__train_mask is None:
            np.random.seed(self.random_state)
            train_ids = np.random.choice(len(data), int(round(self.train_size * len(data))), replace=False)
            train_mask = pd.Series(np.zeros(len(data))).astype(bool)
            train_mask[train_ids] = True
            self.__train_mask = train_mask

        if return_train:
            return data.loc[self.__train_mask, :]
        else:
            return data.loc[~self.__train_mask, :]


class LightningSystem(LightningModule):
    def __init__(self, train_file=None, val_file=None, pred_file=None, test_file=None, random_state=None,
                 model_class=BertClassifier, train_size=0.8, batch_size=32, scheduler_mult=1, learning_rate=0.001,
                 warm_restart=1, num_workers=1, model_params=None, **kwargs):
        super().__init__()

        # apply random seed to torch and numpy
        if random_state:
            seed_everything(seed=random_state, workers=True)

        self.num_workers = num_workers
        self.save_hyperparameters(ignore=["num_workers"] + list(kwargs.keys()))
        self.data_retriever = DataRetriever(
            'sarcastic',
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            pred_file=pred_file,
            random_state=random_state,
            train_size=train_size
        )

        metrics = MetricCollection([Accuracy(num_classes=1, multiclass=False),
                                    Precision(num_classes=1,  multiclass=False),
                                    Recall(num_classes=1,  multiclass=False),
                                    F1Score(num_classes=1, multiclass=False),
                                    AUROC(pos_label=1, num_classes=1)], compute_groups=False)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Get the class constructor from command line options and initialize
        model_class = globals()[model_class]
        # model_params are probably best set in the runner_config.json. Having CLI flags for model_params could get pretty
        # messy.
        if model_params is None:
            self.model = model_class()
        else:
            self.model = model_class(**model_params)
        self.preprocessor = self.model.get_preprocessor()

    def train_dataloader(self):
        print('tng dataloader called')
        train_inputs, train_labels = self.data_retriever.get_train()
        X_train, train_mask = self.preprocessor(train_inputs)
        train_labels = torch.tensor(train_labels.values)

        batch_size = self.hparams.batch_size

        # Create the DataLoader for our training set
        train_data = TensorDataset(X_train, train_mask, train_labels)
        # train_sampler = RandomSampler(train_data)

        print('Training Dataset Size: ', len(train_data))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        print('val dataloader called')

        val_inputs, val_labels = self.data_retriever.get_val()
        X_val, val_mask = self.preprocessor(val_inputs)
        val_labels = torch.tensor(val_labels.values)

        batch_size = self.hparams.batch_size

        # Create the DataLoader for our validation set
        val_data = TensorDataset(X_val, val_mask, val_labels)
        # val_sampler = SequentialSampler(val_data)

        print("Validation Dataset Size: ", len(val_data))
        return DataLoader(val_data, batch_size=batch_size, pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):
        print('test dataloader called')

        test_inputs, test_labels = self.data_retriever.get_test()
        X_test, test_mask = self.preprocessor(test_inputs)
        test_labels = torch.tensor(test_labels.values)

        batch_size = self.hparams.batch_size

        # Create the DataLoader for our validation set
        test_data = TensorDataset(X_test, test_mask, test_labels)
        # val_sampler = SequentialSampler(val_data)

        print("Test Dataset Size: ", len(test_data))
        return DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        pred_inputs = self.data_retriever.get_pred()
        X_pred, pred_mask = self.preprocessor(pred_inputs)

        batch_size = self.hparams.batch_size

        # Create the DataLoader for our validation set
        pred_data = TensorDataset(X_pred, pred_mask)
        return DataLoader(pred_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)

    def forward(self, x, mask):
        return self.model(x, mask)

    def loss(self, label, logit):
        nll = F.cross_entropy(logit, label)
        return nll

    def training_step(self, batch, batch_i):
        x, mask, labels = batch

        logits = self.model.forward(x, mask)
        loss = self.loss(labels, logits)
        preds = torch.argmax(logits, dim=1)

        output_metrics = self.train_metrics(preds, labels)

        self.log("train_step_loss", loss)
        self.log_dict(output_metrics)

        # Need to return a dict and then log in training_step_end if using DP mode.
        # see https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode
        return {"loss":loss, "pred": preds}

    def validation_step(self, batch, batch_i):

        x, mask, labels = batch

        logits = self.model.forward(x, mask)
        loss = self.loss(labels, logits)
        preds = torch.argmax(logits, dim=1)

        output_metrics = self.val_metrics(preds, labels)

        self.log("val_step_loss", loss)
        self.log_dict(output_metrics)

        # Need to return a dict and then log in validation_step_end if using DP mode.
        # see https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode
        return {"loss": loss, "pred": preds}

    def test_step(self, batch, batch_idx):
        x, mask, labels = batch

        logits = self.model.forward(x, mask)
        loss = self.loss(labels, logits)
        preds = torch.argmax(logits, dim=1)

        output_metrics = self.test_metrics(preds, labels)

        self.log("test_step_loss", loss)
        self.log_dict(output_metrics)

        # Need to return a dict and then log in validation_step_end if using DP mode.
        # see https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode
        return {"loss": loss, "logits": logits, "preds": preds, "labels": labels}

    def predict_step(self, batch, batch_idx):
        x, mask = batch
        logits = self.model.forward(x, mask)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def training_epoch_end(self, outputs):
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics = {k + '_epoch': v for k,v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_epoch=True, on_step=False)

        avg_train_loss = torch.mean(torch.stack([output['loss'] for output in outputs]))
        self.log('avg_train_loss', avg_train_loss)


    def validation_epoch_end(self, outputs):
        # Compute first instead of logging so we can add '_epoch' to the metric names
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics = {k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics)

        avg_val_loss = torch.mean(torch.stack([output['loss'] for output in outputs]))
        self.log('avg_val_loss', avg_val_loss)
        print("Validation END ")

    def test_epoch_end(self, outputs):
        # Compute first instead of logging so we can add '_epoch' to the metric names
        epoch_metrics = self.test_metrics.compute()
        epoch_metrics = {k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics)

        # Normally testing only retrieves metrics at the end. This is a little hack to make the logits, labels, and preds
        # Available at the end of testing.
        logits = torch.cat([output['logits'] for output in outputs])
        preds = torch.cat([output['preds'] for output in outputs])
        labels = torch.cat([output['labels'] for output in outputs])
        self.test_results = {'logits': logits, 'preds': preds, 'labels': labels}

        avg_test_loss = torch.mean(torch.stack([output['loss'] for output in outputs]))
        self.log('avg_test_loss', avg_test_loss)

        print("Test END ")

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=1e-8,  # Default epsilon value is 1e-6
                          weight_decay=.01)

        total_steps = self.num_training_steps

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)

       # if self.hparams.warm_restart:
        #    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, hparams.scheduler_freq, T_mult=self.hparams.scheduler_mult, eta_min=0, last_epoch=-1)
       # else:
        #    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.scheduler_freq)

        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        return self.trainer.max_epochs * self.num_training_steps_in_epoch

    @property
    def num_training_steps_in_epoch(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum)

    def get_learn_rate(self):
        if self.trainer is None:
            raise ValueError('A trainer has not been assigned to this LightningModule')
        optimizers, schedulers = self.trainer.optimizers, self.trainer.lr_schedulers
        return schedulers[0]['scheduler'].get_last_lr()[0]

    def get_device_count(self):
        if self.trainer is None:
            raise ValueError('A trainer has not been assigned to this LightningModule')
        nb_devices = 0  # torch.cuda.device_count
        if self.trainer.device_ids is not None:
            nb_devices = len(self.trainer.device_ids)
        return nb_devices


def train(hparams):
    # build new model
    dict_args = vars(hparams)
    model = LightningSystem(**dict_args)
    ckpt_path = None

    if hparams.experiment_version is not None:
        root_path = os.path.join(hparams.log_dir, hparams.experiment_name, f'version_{hparams.experiment_version}')
        last_checkpoint = os.path.join(root_path, 'best_model.ckpt')
        if os.path.exists(last_checkpoint):
            ckpt_path = last_checkpoint

    # If we don't provide experiment version, we need to get it from the logger
    # If we do provide experiment version, we need to set it in the logger.
    # These have reversed orders.
    if hparams.experiment_version is None:
        logger = TensorBoardLogger(hparams.log_dir, hparams.experiment_name, hparams.experiment_version, default_hp_metric=False)
        hparams.experiment_version = f'version_{logger.version}'

    else:
        hparams.experiment_version = f'version_{hparams.experiment_version}'
        logger = TensorBoardLogger(hparams.log_dir, hparams.experiment_name, hparams.experiment_version, default_hp_metric=False)

    model_save_path = os.path.join(hparams.log_dir, hparams.experiment_name, hparams.experiment_version)
    # print(model_save_path)
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path,
        filename="best_model",
        save_top_k=1,
        verbose=True,
        monitor='avg_val_loss',
        mode='min'
    )

    logger.log_hyperparams(hparams, metrics={"avg_train_loss": 1, "avg_val_loss": 1})
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # configure trainer
    print(hparams)

    # Okay, this is pretty dumb. There's enable_checkpoint, checkpoint_callback, and callbacks. The documentation would have you believe that
    # you need to set checkpoint_callback to ModelCheckpoint. It seems checkpoint_callback is being deprecated, so setting it will basically do nothing.
    # enable_checkpoint needs to be set to True and the ModelCheckpoint callback needs to be added to callbacks in Trainer.from_argparse_args.
    # See site-packages/pytorch_lightning/trainer/connectors/callback_connector.py _configure_checkpoint_callbacks
    # (Note: self.trainer.checkpoint_callbacks is just a property that gets all callbacks from Trainer.callbacks that are instances of ModelCheckpoint
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        default_root_dir=hparams.log_dir,
        enable_checkpointing=True,
        callbacks=[checkpoint, lr_monitor],
        resume_from_checkpoint=ckpt_path
    )

    trainer.fit(model)


def test(hparams):
    """
    Run testing and evaluate performance metrics on model output. Writes model output and metrics to file.
    """

    root_path = os.path.join(hparams.log_dir, hparams.experiment_name, f'version_{hparams.experiment_version}')
    last_checkpoint = os.path.join(root_path, 'best_model.ckpt')
    if not os.path.exists(last_checkpoint):
        raise ValueError(f"Could not load model given dir, name, and version. File does not exist: {last_checkpoint}")

    hparam_file = os.path.join(root_path, 'hparams.yaml')
    model = LightningSystem.load_from_checkpoint(last_checkpoint, test_file=hparams.test_file, hparams_file=hparam_file,
                                                 batch_size=hparams.batch_size, num_workers=hparams.num_workers)

    experiment_name = os.path.join(hparams.experiment_name, f'version_{hparams.experiment_version}', 'testing')
    logger = TensorBoardLogger(hparams.log_dir, hparams.experiment_name, f'version_{hparams.experiment_version}',
                               sub_dir='testing', default_hp_metric=False)

    logger.log_hyperparams(hparams, metrics={"avg_test_loss": 10})

    # configure trainer
    print(hparams)

    trainer = Trainer.from_argparse_args(hparams, logger=logger, default_root_dir=hparams.log_dir,
                                         enable_checkpointing=False)

    metrics = trainer.test(model)[0]

    # We don't need the last step loss, so remove it
    del metrics['test_step_loss']

    # Write predictions, logits, and labels to file
    output_dict = model.test_results
    logits = output_dict.pop('logits')
    output_dict['logits_0'] = logits[:,0]
    output_dict['logits_1'] = logits[:, 1]

    # move output dict items to cpu 
    for k, v in output_dict.items():
      output_dict[k] = v.cpu()
    
    outputs = pd.DataFrame(output_dict)
    outputs.to_csv(hparams.pred_output_file, index=False)

    report = classification_report(outputs["labels"], outputs["preds"], zero_division=0)
    f = open(hparams.metrics_output_file, "w")
    print(report, file=f)
    f.close()
    # Write metrics to file
    # metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    # metrics_df.index.name = 'Metric Name'
    # metrics_df.to_csv(hparams.metrics_output_file)


def predict(hparams):
    root_path = os.path.join(hparams.log_dir, hparams.experiment_name, f'version_{hparams.experiment_version}')
    last_checkpoint = os.path.join(root_path, 'best_model.ckpt')
    if not os.path.exists(last_checkpoint):
        raise ValueError(f"Could not load model given dir, name, and version. File does not exist: {last_checkpoint}")
    output_file = hparams.output_file

    model = LightningSystem.load_from_checkpoint(last_checkpoint, pred_file=hparams.predict_file, batch_size=hparams.batch_size, num_workers=hparams.num_workers)
    trainer = Trainer.from_argparse_args(hparams)
    outputs = trainer.predict(model)

    logits, preds = zip(*outputs)
    preds = torch.cat(preds)
    logits = torch.cat(logits)

    df = pd.DataFrame(preds, columns=['prediction'])
    df[['logit_0', 'logit_1']] = logits
    df.to_csv(hparams.output_file, sep=',', index=False)


def convert_saved_model_to_checkpoint(hparams):
    model_save_path = hparams.model_save_path
    del hparams.model_save_path
    print(hparams)

    with open(model_save_path, 'rb') as f:
        model = torch.load(f, map_location=torch.device('cpu'))
    dict_args = vars(hparams)
    lightning_model = LightningSystem(**dict_args, train_file='data/balanced_train_En.csv')
    lightning_model.model = model

    if hparams.experiment_version is None:
        logger = TensorBoardLogger(hparams.log_dir, hparams.experiment_name, hparams.experiment_version, default_hp_metric=False)
        hparams.experiment_version = f'version_{logger.version}'

    else:
        hparams.experiment_version = f'version_{hparams.experiment_version}'
        logger = TensorBoardLogger(hparams.log_dir, hparams.experiment_name, hparams.experiment_version, default_hp_metric=False)


    model_save_path = os.path.join(hparams.log_dir, hparams.experiment_name, hparams.experiment_version)
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path,
        filename="best_model",
        save_top_k=1,
        verbose=True
    )

    logger.log_hyperparams(hparams, metrics={"avg_train_loss": 10, "avg_val_loss": 10})

    trainer = Trainer.from_argparse_args(hparams, logger=logger, default_root_dir=hparams.log_dir,
                      enable_checkpointing=True, callbacks=[checkpoint])

    trainer.validate(lightning_model)

    trainer.save_checkpoint(os.path.join(model_save_path, "best_model.ckpt"))


def add_program_args(parser):
    parser.add_argument('--log-dir', default=".logging/", type=str, help='Folder to use for logging experiments.')
    parser.add_argument('--experiment-name', default='default', type=str, help='name of experiment to log.')
    parser.add_argument('--experiment-version', default=None, type=str, help='Version of experiment to log.')
    parser.add_argument('--accelerator', choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'auto'], default='auto', help='Run computations on cpu, gpu, or other accelerators.')
    parser.add_argument('--devices', default=1, type=int, help='Number of devices to use for computations.')
    parser.add_argument('--batch-size', default=32, type=int, choices=[8, 16, 32, 64, 128],
                               help='batch size will be divided over all gpus being used across all nodes.')
    parser.add_argument('--num-workers', default=-1, type=int, help="Num cpu cores to use for Dataloading")
    parser.add_argument('-c', '--config', default=None, type=argparse.FileType('r'), help='Config file for options. Will be overridden by cli flags.')
    parser.add_argument('--model-class', default='BertClassifier', type=str, help="Name of model class to use for modeling.")


def add_train_args(parser):
    parser.add_argument('train_file', type=str, help='Path to training file')
    parser.add_argument('--val-file', type=str, default=None, help='Path to validation file.')
    parser.add_argument('--train-size', default=0.8, type=float, help='Pct of train_file to use as training data. Rest will be validation data.')
    parser.add_argument('--fast-dev-run', action='store_true', help='Sets trainer to debug mode which will run 1 batch of training and 1 batch of validation.')
    parser.add_argument('--disable-checkpoints', action='store_true', help='Turns off experiment checkpointing. Useful when debugging')
    parser.add_argument('--description', default='no description', type=str, help='Provide a description for the experiment.')
    parser.add_argument('--random-state', default=None, type=int, help='Random seed')
    parser.add_argument('--log-every-n-steps', type=int, default=5, help='Log every n training or validation steps')
    parser.add_argument('--max-steps', type=int, default=None, help="Max number of steps")
    parser.add_argument('--max-epochs', type=int, default=None, help="Max number of epochs to train on")
    parser.add_argument('--max-time', type=int, default=None)
    parser.add_argument('--distributed-backend', default='dp', choices=['dp', 'ddp'], type=str, help='Paralllel backend to use.')
    parser.add_argument('--limit-train-batches', default=1.0, type=float, help='Percent of training dataset to check.')
    parser.add_argument('--val-check-interval', default=1.0, type=float, help='Percent interval of training epoch to check validation.')
    parser.add_argument('--limit-val-batches', default=1.0, type=float, help='Percentage of validation dataset to check.')
    parser.add_argument('--gradient-clip-val', default=3.0, type=float, help='Clips the gradient norm to guard against exploding gradients.')

    add_model_specific_args(parser)


def add_test_args(parser):
    parser.add_argument('test_file', action='store', type=str)
    parser.add_argument('pred_output_file', action='store',type=str)
    parser.add_argument('metrics_output_file', action='store', type=str)


def add_predict_args(parser):
    parser.add_argument('predict_file', action='store', type=str)
    parser.add_argument('output_file', action='store',type=str)


def add_convert_args(parser):
    parser.add_argument('model_save_path', action='store', type=str, help='Path to saved model.')


def add_model_specific_args(parser):
    parser.add_argument('--warm-restart', action='store_true', help='Use warm restarts')
    parser.add_argument('--track-grad-norm', default=-1, type=int, choices=[-1, 0, 1, 2], help='Log the norm of the parameter gradients')
    parser.add_argument('--scheduler-freq', default=10, type=int, help='Number of epochs for cosine annealing.')
    parser.add_argument('--scheduler-mult', default=1, choices=range(1, 6), type=int, help='Multiplier for cosine annealing with warm restarts.')
    parser.add_argument('--learning-rate', default=0.001, type=float, choices=[0.0001, 0.0005, 0.001, 0.005, 0.01])


if __name__ == '__main__':
    global_parser = argparse.ArgumentParser(add_help=False)
    add_program_args(global_parser)
    parser = argparse.ArgumentParser(add_help=True)

    subparser = parser.add_subparsers(dest='subparser_name', required=True)
    train_parser = subparser.add_parser('train', parents=[global_parser], add_help=True, help='Build a model from the input file.')
    test_parser = subparser.add_parser('test', parents=[global_parser], add_help=True, help='Run testing on input file using a model.')
    predict_parser = subparser.add_parser('predict', parents=[global_parser], add_help=True, help='predict input file using a model.')
    convert_parser = subparser.add_parser('convert', parents=[global_parser], add_help=True, help='Convert saved model to checkpoint')
    add_train_args(train_parser)
    add_test_args(test_parser)
    add_convert_args(convert_parser)
    add_predict_args(predict_parser)

    args = sys.argv[1:]
    hparams = parser.parse_args(args)

    # If config file is provided, set parser defaults from config file and reparse options
    if hparams.config is not None:
        config = json.load(hparams.config)
        train_parser.set_defaults(**config)
        test_parser.set_defaults(**config)
        predict_parser.set_defaults(**config)
        convert_parser.set_defaults(**config)
        hparams = parser.parse_args(args)
        hparams.config = None

    if hparams.num_workers < 1:
        hparams.num_workers = os.cpu_count()

    if hparams.subparser_name == 'train':
        train(hparams)

    elif hparams.subparser_name == 'test':
        test(hparams)

    elif hparams.subparser_name == 'predict':
        predict(hparams)

    else:
        convert_saved_model_to_checkpoint(hparams)

"""
NOTE: To view logs and visualizations run:
tensorboard --logdir=ling573-2022-spring/ .logging/
"""
