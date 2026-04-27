import os
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.distributed as dist

from modules.metrics.metrics import compute_coco_scores
from utils.utils import read_json_file, write_json_file


class BaseReportModel(pl.LightningModule, ABC):

    def __init__(self, args, model, tokenizer):
        super().__init__()
        self.model = model
        for parameter in self.model.parameters():
            if not parameter.is_contiguous():
                parameter.data = parameter.data.contiguous()

        self.tokenizer = tokenizer
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.lr_patience = args.lr_patience
        self.visualize_batch = getattr(args, 'visualize_batch', 0)
        self.output_dir = args.output_dir
        self.predictions = {}

        reports = read_json_file(args.reports_json_path)
        self.reports = {
            report['id']: report['report'] for split in reports for report in reports[split]
        }

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        trainable_params = filter(lambda parameter: parameter.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.lr_patience,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_validation_epoch_end(self):
        self.log_metrics('val', compute_coco_scores, True)
        self.predictions.clear()

    def on_test_epoch_end(self):
        self.log_metrics('test', compute_coco_scores, True)
        if self.trainer.is_global_zero:
            self.write_predictions()
        self.predictions.clear()

    def save_predictions_from_ids(self, slide_ids, pred_ids, target_ids):
        pred_texts = self.tokenizer.decode_batch(pred_ids)
        target_texts = self.tokenizer.decode_batch(target_ids)
        self.save_prediction_texts(slide_ids, pred_texts, target_texts)
        return pred_texts, target_texts

    def save_prediction_texts(self, slide_ids, pred_texts, target_texts):
        for index, slide_id in enumerate(slide_ids):
            self.predictions[slide_id] = {
                'pred': pred_texts[index],
                'target': target_texts[index],
            }

    def should_visualize(self, batch_idx):
        return bool(self.visualize_batch) and batch_idx % self.visualize_batch == 0

    def print_results(self, slide_ids, pred_texts, target_texts):
        red = '\033[91m'
        reset = '\033[0m'
        blue = '\033[94m'

        for index, slide_id in enumerate(slide_ids):
            ground_truth = self.reports.get(slide_id, target_texts[index])
            print('*' * 100)
            print(f'{red} Predicted report for slide: {slide_id}: {pred_texts[index]} {reset}')
            print(f'Ground truth: {target_texts[index]}')
            print(f'{blue} Ground truth: {ground_truth} {reset}')
            print('*' * 100)

    def write_predictions(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        write_json_file(self.gather_predictions(), f'{self.output_dir}/predictions.json')

    def log_metrics(self, stage, evaluate_fn, prog_bar):
        predictions = self.gather_predictions()
        pred_texts = []
        target_texts = []
        for slide_id, prediction in predictions.items():
            pred_texts.append(prediction['pred'])
            target_texts.append(prediction['target'])

        metrics = evaluate_fn(list(zip(pred_texts, target_texts)))

        for metric_name, metric_score in metrics.items():
            self.log(
                f'{stage}_{metric_name}',
                metric_score,
                on_epoch=True,
                prog_bar=prog_bar,
                sync_dist=True,
            )

    def gather_predictions(self):
        if not dist.is_available() or not dist.is_initialized():
            return dict(self.predictions)

        gathered_predictions = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_predictions, dict(self.predictions))

        merged_predictions = {}
        for rank_predictions in gathered_predictions:
            merged_predictions.update(rank_predictions)
        return merged_predictions
