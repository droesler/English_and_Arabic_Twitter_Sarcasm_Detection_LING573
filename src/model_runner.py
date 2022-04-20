import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers.optimization import AdamW
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score
from models import BertClassifier, BertSmallClassifier



class DataRetriever:
    def __init__(self, label_to_classify=None, train_file=None, val_file=None, pred_file=None, random_state = None, train_size=0.8):
        if random_state is None or np.isnan(random_state):
            random_state = np.random.randint(0, 1e8)
        self.random_state = int(random_state)
        self.label_to_classify = label_to_classify
        self.__train_mask = None
        self.train_size = train_size
        self.train_file = train_file
        self.val_file = val_file
        self.pred_file = pred_file


    def get_train(self, delimiter=','):
        if self.val_file is None:
            train_data = self._split_train_val(return_train=True)
        else:
            train_data = pd.read_csv(self.train_file, sep=delimiter)
        train_data = train_data.loc[train_data['tweet'].notna(), :]
        return train_data['tweet'], train_data[self.label_to_classify]

    def get_val(self, delimiter=','):
        if self.val_file is None:
            val_data = self._split_train_val(return_train=False)
        else:
            val_data = pd.read_csv(self.val_file, sep=delimiter)
        val_data = val_data.loc[val_data['tweet'].notna(), :]
        return val_data['tweet'], val_data[self.label_to_classify]

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
    
    def __init__(self, train_file=None, val_file=None, pred_file=None, random_state=None, train_size=0.8, batch_size=32, scheduler_mult=1, learning_rate=0.001, warm_restart=1, num_workers=1, **kwargs):
        super().__init__()

        self.num_workers = num_workers
        self.save_hyperparameters(ignore=["num_workers"])


        self.data_retriever = DataRetriever('sarcastic', train_file=train_file, val_file=val_file, pred_file=pred_file,
                                       random_state=random_state, train_size=train_size)

        metrics = MetricCollection([Accuracy(),
                                    Precision(),
                                    Recall(),
                                    F1Score(),
                                    AUROC(pos_label=1, num_classes=1)], compute_groups=False)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

        self.model = BertSmallClassifier()
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

    def training_step(self, data_batch, batch_i):
        x, mask, labels = data_batch

        logits = self.model.forward(x, mask)
        loss = self.loss(labels, logits)
        preds = torch.argmax(logits, dim=1)

        output_metrics = self.train_metrics(preds, labels)

        self.log("train_step_loss", loss)
        self.log_dict(output_metrics)

        # Need to return a dict and then log in training_step_end if using DP mode.
        # see https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode
        return {"loss":loss, "pred": preds}

    def validation_step(self, data_batch, batch_i):

        x, mask, labels = data_batch

        logits = self.model.forward(x, mask)
        loss = self.loss(labels, logits)
        preds = torch.argmax(logits, dim=1)

        output_metrics = self.val_metrics(preds, labels)

        self.log("val_step_loss", loss)
        self.log_dict(output_metrics)

        # Need to return a dict and then log in validation_step_end if using DP mode.
        # see https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode
        return {"loss":loss, "pred": preds}

    def predict_step(self, batch, batch_idx):
        x, mask = batch
        logits = self.model.forward(x, mask)
        preds = torch.argmax(logits, dim=1)
        return logits, preds

    def training_epoch_end(self, outputs):
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics = {k + '_epoch': v for k,v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_epoch=True, on_step=False)

        avg_train_loss = np.mean([output['loss'] for output in outputs])
        self.log('avg_train_loss', avg_train_loss)

    def validation_epoch_end(self, outputs):
        # Compute first instead of logging so we can add '_epoch' to the metric names
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics = {k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics)

        avg_val_loss = np.mean([output['loss'] for output in outputs])
        self.log('avg_val_loss', avg_val_loss)
        print("Validation END ")


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=1e-8,  # Default epsilon value is 1e-6
                          weight_decay=.01)

        if self.hparams.warm_restart:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, hparams.scheduler_freq, T_mult=self.hparams.scheduler_mult, eta_min=0, last_epoch=-1)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.scheduler_freq)

        return [optimizer], [scheduler]

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
        #julie here -- adjusting the code to see if it resolves a syntax error that I'm getting on my end
        v='version_{}'
        vrsn=v.format(hparams.experiment_version)
        root_path=os.path.join(hparams.log_dir,hparams.experiment_name,vrsn)
        #root_path = os.path.join(hparams.log_dir, hparams.experiment_name, f'version_{hparams.experiment_version}')
        last_checkpoint = os.path.join(root_path, 'best_model.ckpt')
        if os.path.exists(last_checkpoint):
            ckpt_path = last_checkpoint

    # If we don't provide experiment version, we need to get it from the logger
    # If we do provide experiment version, we need to set it in the logger.
    # These have reversed orders.
    if hparams.experiment_version is None:
        logger = TensorBoardLogger(hparams.log_dir, hparams.experiment_name, hparams.experiment_version, default_hp_metric=False)
        #julie here again -- 
        v='version_{}'
        vrsn=v.format(logger.version)
        haparams.experiment_version=vrsn
        #hparams.experiment_version = f'version_{logger.version}'
    else:
        hparams.experiment_version = f'version_{hparams.experiment_version}'
        logger = TensorBoardLogger(hparams.log_dir, hparams.experiment_name, hparams.experiment_version, default_hp_metric=False)


    model_save_path = os.path.join(hparams.log_dir, hparams.experiment_name, hparams.experiment_version)
    #print(model_save_path)
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path,
        filename="best_model",
        save_top_k=1,
        verbose=True,
        monitor='avg_val_loss',
        mode='min'
    )

    logger.log_hyperparams(hparams, metrics={"avg_train_loss": 1, "avg_val_loss": 1})

    #configure trainer
    print(hparams)

    # Okay, this is pretty dumb. There's enable_checkpoint, checkpoint_callback, and callbacks. The documentation would have you believe that
    # you need to set checkpoint_callback to ModelCheckpoint. It seems checkpoint_callback is being deprecated, so setting it will basically do nothing.
    # enable_checkpoint needs to be set to True and the ModelCheckpoint callback needs to be added to callbacks in Trainer.from_argparse_args.
    # See site-packages/pytorch_lightning/trainer/connectors/callback_connector.py _configure_checkpoint_callbacks
    # (Note: self.trainer.checkpoint_callbacks is just a property that gets all callbacks from Trainer.callbacks that are instances of ModelCheckpoint
    trainer = Trainer.from_argparse_args(hparams, logger=logger, default_root_dir=hparams.log_dir,
                      enable_checkpointing=True, callbacks=[checkpoint], resume_from_checkpoint=ckpt_path)

    trainer.fit(model)


def score(hparams):
    root_path = os.path.join(hparams.log_dir, hparams.experiment_name, f'version_{hparams.experiment_version}')
    last_checkpoint = os.path.join(root_path, 'best_model.ckpt')
    if not os.path.exists(last_checkpoint):
        raise ValueError(f"Could not load model given dir, name, and version. File does not exist: {last_checkpoint}")
    output_file = hparams.output_file

    model = LightningSystem.load_from_checkpoint(last_checkpoint, pred_file=hparams.score_file, batch_size=hparams.batch_size, num_workers=hparams.num_workers)
    trainer = Trainer.from_argparse_args(hparams)
    outputs = trainer.predict(model)

    logits, preds = zip(*outputs)
    preds = torch.cat(preds)
    logits = torch.cat(logits)

    df = pd.DataFrame(preds, columns=['prediction'])
    df[['logit_0', 'logit_1']] = logits
    df.to_csv(hparams.output_file, sep=',', index=False)

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


def add_train_args(parser):
    parser.add_argument('train_file', type=str, help='Path to training file')
    parser.add_argument('--val-file', type=str, default=None, help='Path to validation file.')
    parser.add_argument('--train-size', default=0.8, type=float, help='Pct of train_file to use as training data. Rest will be validation data.')
    parser.add_argument('--fast-dev-run', action='store_true', help='Sets trainer to debug mode which will run 1 batch of training and 1 batch of validation.')
    parser.add_argument('--disable-checkpoints', action='store_true', help='Turns off experiment checkpointing. Useful when debugging')
    parser.add_argument('--description', default='no description', type=str, help='Provide a description for the experiment.')
    parser.add_argument('--random-state', default=None, type=int, help='Random seed')
    parser.add_argument('--log-every-n-steps', type=int, default=5, help='Log every n training or validation steps')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--max-time', type=int, default=None)
    parser.add_argument('--distributed-backend', default='dp', choices=['dp', 'ddp'], type=str, help='Paralllel backend to use.')
    parser.add_argument('--nb-epoch', default=10, type=int, help='Number of epochs to train on.')
    parser.add_argument('--limit-train-batches', default=1.0, type=float, help='Percent of training dataset to check.')
    parser.add_argument('--val-check-interval', default=1.0, type=float, help='Percent interval of training epoch to check validation.')
    parser.add_argument('--limit-val-batches', default=1.0, type=float, help='Percentage of validation dataset to check.')
    parser.add_argument('--gradient-clip-val', default=3.0, type=float, help='Clips the gradient norm to guard against exploding gradients.')

    add_model_specific_args(parser)

def add_score_args(parser):
    parser.add_argument('score_file', action='store', type=str)
    parser.add_argument('output_file', action='store',type=str)

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
    score_parser = subparser.add_parser('score', parents=[global_parser], add_help=True, help='Score input file using a model.')
    add_train_args(train_parser)
    add_score_args(score_parser)

    args = sys.argv[1:]
    hparams = parser.parse_args(args)

    # If config file is provided, set parser defaults from config file and reparse options
    if hparams.config is not None:
        config = json.load(hparams.config)
        train_parser.set_defaults(**config)
        score_parser.set_defaults(**config)
        hparams = parser.parse_args(args)
        hparams.config = None
    
    if hparams.num_workers < 1:
        hparams.num_workers = os.cpu_count()
    
    if hparams.subparser_name == 'score':
        score(hparams)
    
    else:
        train(hparams)

"""
NOTE: To view logs and visualizations run:
tensorboard --logdir=ling573-2022-spring/ .logging/
"""
