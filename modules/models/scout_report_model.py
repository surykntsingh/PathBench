import torch

from modules.loss import LanguageModelCriterion
from modules.models.base_report_model import BaseReportModel


class ScoutReportModel(BaseReportModel):

    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)
        self.g_lambda = args.g_lambda

    def get_attn_regularization(self, weights, eps=1e-8):
        entropy = -(weights * (weights + eps).log()).sum(dim=-1)
        return entropy.mean()

    def loss_fn(self, output, reports_ids, reports_masks, attns):
        language_criterion = LanguageModelCriterion()
        caption_loss = language_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        attn_reg = self.get_attn_regularization(attns)
        return caption_loss + self.g_lambda * attn_reg

    def training_step(self, batch, batch_idx):
        _, features, report_ids, report_masks = batch
        output, attn = self.model(features, report_ids, mode='train')
        loss = self.loss_fn(output, report_ids, report_masks, attn)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch

        output, attn = self.model(features, report_ids, mode='train')
        loss = self.loss_fn(output, report_ids, report_masks, attn)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        pred_ids, attn = self.model(features, report_ids, mode='sample')
        pred_texts, ground_truths = self.save_predictions_from_ids(
            slide_ids,
            pred_ids.detach().cpu().numpy(),
            report_ids[:, 1:].detach().cpu().numpy(),
        )

        if self.should_visualize(batch_idx):
            self.print_results(slide_ids, pred_texts, ground_truths)
            self.visualize_attn(attn)

    def test_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch

        output, attn = self.model(features, report_ids, mode='train')
        loss = self.loss_fn(output, report_ids, report_masks, attn)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        pred_ids, attn = self.model(features, report_ids, mode='sample')
        pred_texts, ground_truths = self.save_predictions_from_ids(
            slide_ids,
            pred_ids.detach().cpu().numpy(),
            report_ids[:, 1:].detach().cpu().numpy(),
        )

        if self.should_visualize(batch_idx):
            self.print_results(slide_ids, pred_texts, ground_truths)
            self.visualize_attn(attn)

    def predict_step(self, batch, batch_idx):
        slide_ids, features = batch
        pred_ids, _ = self.model(features, mode='sample')
        pred_texts = self.tokenizer.decode_batch(pred_ids.detach().cpu().numpy())
        target_texts = [self.reports[slide_id] for slide_id in slide_ids]
        self.print_results(slide_ids, pred_texts, target_texts)
        return slide_ids, pred_texts

    def visualize_attn(self, weights):
        weights = weights.detach().cpu()
        w_patch = weights[:, :, :, 0].mean().numpy()
        w_slide = weights[:, :, :, 1].mean().numpy()
        w_concept = weights[:, :, :, 2].mean().numpy()
        print(f'w_patch: {w_patch}, w_slide: {w_slide}, w_concept: {w_concept}')
