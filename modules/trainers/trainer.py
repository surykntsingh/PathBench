import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner


class Trainer:

    def __init__(self, args, tokenizer):
        self.best_model_path = None
        self.ckpt_path = f'{args.output_dir}/ckpt'
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.max_epochs = args.max_epochs
        # pl.seed_everything(42)
        # torch.set_float32_matmul_precision('high')
        # torch.use_deterministic_algorithms(True)
        self.trainer = None
        self.devices = args.devices if type(args.devices)==int else list(map(int, args.devices.split(',')))
        self.args = args
        self.tokenizer = tokenizer
        self.strategy = 'ddp_find_unused_parameters_true' if args.unused_params else 'ddp'

    def train(self, model, datamodule, fast_dev_run=False):
        monitor_metric = f"val_{self.args.monitor_metric}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_path,  # Directory to save checkpoints
            filename="best_model_{epoch:02d}_{val_loss:.5f}_{"+monitor_metric+":.5f}",  # Naming convention
            monitor=monitor_metric,  # Metric to monitor for saving best checkpoints
            mode=self.args.monitor_mode,  # Whether to minimize or maximize the monitored metric
            save_top_k=1,  # Number of best checkpoints to keep
            save_last=True  # Save the last checkpoint regardless of the monitored metric
        )
        early_stop_callback = EarlyStopping(monitor=f"val_{self.args.monitor_mode}", min_delta=1e-5,
                                            patience=self.args.early_stop, verbose=True, mode="max")

        # suggested_lr = self.find_lr(model, datamodule)
        # print(f'setting lr: {suggested_lr}')
        # # Set suggested LR
        # model.hparams.lr = suggested_lr

        self.trainer = pl.Trainer(
            # precision="bf16-mixed",
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback,early_stop_callback],
            accelerator='gpu',
            devices=self.devices,
            strategy=self.strategy,
            enable_progress_bar=True,
            log_every_n_steps=1,
            fast_dev_run=fast_dev_run
        )

        self.trainer.fit(
            model, datamodule=datamodule
        )

        train_metrics = self.trainer.logged_metrics
        self.best_model_path = checkpoint_callback.best_model_path
        # self.best = train_metrics['val_loss']
        return train_metrics, self.trainer

    def find_lr(self, model, datamodule):
        trainer = pl.Trainer(
            # precision="16-mixed",
            accelerator='gpu',
            devices=self.devices,
            strategy=self.strategy,
            enable_progress_bar=True
        )
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=datamodule)

        suggested_lr = lr_finder.suggestion()
        print(f'suggested lr: {suggested_lr}')

        return suggested_lr


    def test(self, model, datamodule, fast_dev_run=False):

        trainer = pl.Trainer(
            # precision="bf16-mixed",
            accelerator='gpu',
            devices=self.devices,
            strategy=self.strategy,
            enable_progress_bar=True,
            log_every_n_steps=1,
            fast_dev_run=fast_dev_run
        )

        trainer.test(
            model, datamodule=datamodule
        )
        test_metrics = trainer.logged_metrics
        return test_metrics, trainer

    def predict(self, model, datamodule, fast_dev_run=False):

        trainer = pl.Trainer(
            # precision="16-mixed",
            accelerator='gpu',
            devices=self.devices,
            strategy=self.strategy,
            enable_progress_bar=True,
            log_every_n_steps=1,
            fast_dev_run=fast_dev_run
        )

        preds = trainer.predict(
            model, datamodule=datamodule
        )
        return preds

