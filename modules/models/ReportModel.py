import json
import gc
import os
import torch
import torch.distributed as dist
import pytorch_lightning as pl

from modules.loss import LanguageModelCriterion
from modules.metrics.metrics import compute_scores, compute_coco_scores
# from tokenizers.report_tokenizers import Tokenizer
# from modules.report_gen_model import ReportGenModel
from utils.utils import read_json_file, write_json_file


class ReportModel(pl.LightningModule):

    def __init__(self, args,model, tokenizer):
        super().__init__()
        self.model = model
        # self.model.tie_weights()
        # self.concept_lambda = args.concept_lambda
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        self.tokenizer = tokenizer
        self.learning_rate = args.lr
        self.__weight_decay = args.weight_decay
        self.__lr_patience =args.lr_patience
        self.__g_lambda = args.g_lambda

        self.predictions = {}

        reports = read_json_file(args.reports_json_path)
        # self.reports = {
        #     report['id'].split('.')[0]: report['report']  for split in reports for report in reports[split]
        # }
        self.reports = {
            report['id']: report['report'] for split in reports for report in reports[split]
        }

        self.__output_dir = args.output_dir


    def get_attn_regularization(self, weights, eps=1e-8):
        # Attention Regularization

        entropy = - (weights * (weights + eps).log()).sum(dim=-1)  # [B, L, H]
        return entropy.mean()

    def loss_fn(self, output, reports_ids, reports_masks, attns):
        language_criterion = LanguageModelCriterion()
        caption_loss = language_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()

        attn_reg = self.get_attn_regularization(attns)

        total_loss = caption_loss + self.__g_lambda * attn_reg
        return total_loss


    def training_step(self, batch, batch_idx):
        # gc.collect()
        _, features, report_ids, report_masks = batch
        # print(f'train features: {features}')
        output,attn = self.model(features, report_ids, mode='train')

        loss = self.loss_fn(output, report_ids, report_masks, attn)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        del output
        return loss

    def validation_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch
        with torch.no_grad():
            output_,attn = self.model(features, report_ids, mode='train')
            loss = self.loss_fn(output_, report_ids, report_masks, attn)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            # del output_
            # torch.cuda.empty_cache()


        with torch.no_grad():
            output, attn = self.model(features, report_ids, mode='sample')
            self.__visualize_attn(attn)
            output = output.detach().cpu().numpy()
            pred_texts = self.tokenizer.decode_batch(output)
            ground_truths = self.tokenizer.decode_batch(report_ids[:, 1:].cpu().numpy())
            self.__save_predictions(slide_ids, pred_texts, ground_truths)
            target_texts = [self.reports[slide_id] for slide_id in slide_ids]
        if batch_idx % 10 == 0:
            self.__print_results(slide_ids, pred_texts, ground_truths)



    def test_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch

        with torch.no_grad():
            output_,attn  = self.model(features, report_ids, mode='train')
            loss = self.loss_fn(output_, report_ids, report_masks, attn)
            self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            del output_
            torch.cuda.empty_cache()

        with torch.no_grad():
            output,attn = self.model(features, report_ids, mode='sample')
            output = output.detach().cpu().numpy()
            self.__visualize_attn(attn)
            pred_texts = self.tokenizer.decode_batch(output)
            target_texts = [self.reports[slide_id] for slide_id in slide_ids]
            ground_truths = self.tokenizer.decode_batch(report_ids[:, 1:].cpu().numpy())
            self.__save_predictions(slide_ids, pred_texts, ground_truths)
            if batch_idx % 10 == 0:
                self.__print_results(slide_ids, pred_texts, ground_truths)


    def predict_step(self, batch, batch_idx):
        slide_ids, features = batch
        with torch.no_grad():
            output,concept_attn_maps = self.model(features, mode='sample')
        pred_texts = self.tokenizer.decode_batch(output.detach().cpu().numpy())
        target_texts = [self.reports[slide_id] for slide_id in slide_ids]

        self.__print_results(slide_ids, pred_texts, target_texts)

        del output
        return slide_ids,pred_texts

    def on_validation_epoch_end(self):
        self.__log_metrics('val', compute_coco_scores, True)
        self.predictions.clear()

    def on_test_epoch_end(self):
        self.__log_metrics('test', compute_coco_scores, True)
        if self.trainer.is_global_zero:
            self.__write_predictions()
        self.predictions.clear()

    def configure_optimizers(self):
        d_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(d_params, lr=self.learning_rate, weight_decay=self.__weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.__lr_patience)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def __print_results(self, slide_ids, pred_texts, target_texts):
        RED = '\033[91m'
        RESET = '\033[0m'
        BLUE = '\033[94m'

        for i in range(len(slide_ids)):
            ground_truth = self.reports[slide_ids[i]]

            print('*' * 100)
            print(f'{RED} Predicted report for slide: {slide_ids[i]}: {pred_texts[i]} {RESET}')
            print(f'Ground truth: {target_texts[i]}')
            print(f'{BLUE} Ground truth: {ground_truth} {RESET}')

            # json_string = json.dumps(extract_fields(pred_text), indent=4)
            # print(f'{RED} {json_string} {RESET}')
            print('*' * 100)

    def __save_predictions(self, slide_ids, pred_texts, ground_truths):
        # print(f'slide_ids: {slide_ids}, pred_texts: {pred_texts}')
        for i, slide_id in enumerate(slide_ids):
            self.predictions[slide_id] = {
                'pred': pred_texts[i],
                'target': ground_truths[i]
            }

    def __write_predictions(self):
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir, exist_ok=True)
        write_json_file(self.__gather_predictions(), f'{self.__output_dir}/predictions.json')


    def __log_metrics(self, stage, evaluate_fn, prog_bar):
        predictions = self.__gather_predictions()
        pred_texts = []
        target_texts = []
        for slide_id in predictions:
            pred_texts.append(predictions[slide_id]['pred'])
            target_texts.append(predictions[slide_id]['target'])

        metrics = evaluate_fn(list(zip(pred_texts, target_texts)))

        for metric_name, metric_score in metrics.items():
            self.log(
                f'{stage}_{metric_name}', metric_score, on_epoch=True, prog_bar=prog_bar, sync_dist=True
            )

    def __gather_predictions(self):
        if not dist.is_available() or not dist.is_initialized():
            return dict(self.predictions)

        gathered_predictions = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_predictions, dict(self.predictions))

        merged_predictions = {}
        for rank_predictions in gathered_predictions:
            merged_predictions.update(rank_predictions)
        return merged_predictions

    def __visualize_attn(self, weights):
        weights = weights.detach().cpu()
        w_patch = weights[:, :, :, 0].mean().numpy()
        w_slide = weights[:, :, :, 1].mean().numpy()
        w_concept = weights[:, :, :, 2].mean().numpy()

        print(f'w_patch: {w_patch}, w_slide: {w_slide}, w_concept: {w_concept}')
