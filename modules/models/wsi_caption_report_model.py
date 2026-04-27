from modules.loss import LanguageModelCriterion
from modules.models.base_report_model import BaseReportModel


class WSICaptionReportModel(BaseReportModel):

    def loss_fn(self, output, reports_ids, reports_masks):
        language_criterion = LanguageModelCriterion()
        return language_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    def get_encoder_features(self, features):
        return features['patch']

    def training_step(self, batch, batch_idx):
        _, features, report_ids, report_masks = batch
        output = self.model(self.get_encoder_features(features), report_ids, mode='train')
        loss = self.loss_fn(output, report_ids, report_masks)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch

        output = self.model(self.get_encoder_features(features), report_ids, mode='train')
        loss = self.loss_fn(output, report_ids, report_masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        pred_ids = self.model(self.get_encoder_features(features), report_ids, mode='sample')
        pred_texts, ground_truths = self.save_predictions_from_ids(
            slide_ids,
            pred_ids.detach().cpu().numpy(),
            report_ids[:, 1:].detach().cpu().numpy(),
        )

        if self.should_visualize(batch_idx):
            self.print_results(slide_ids, pred_texts, ground_truths)

    def test_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch

        output = self.model(self.get_encoder_features(features), report_ids, mode='train')
        loss = self.loss_fn(output, report_ids, report_masks)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        pred_ids = self.model(self.get_encoder_features(features), report_ids, mode='sample')
        pred_texts, ground_truths = self.save_predictions_from_ids(
            slide_ids,
            pred_ids.detach().cpu().numpy(),
            report_ids[:, 1:].detach().cpu().numpy(),
        )

        if self.should_visualize(batch_idx):
            self.print_results(slide_ids, pred_texts, ground_truths)

    def predict_step(self, batch, batch_idx):
        slide_ids, features = batch
        pred_ids = self.model(self.get_encoder_features(features), mode='sample')
        pred_texts = self.tokenizer.decode_batch(pred_ids.detach().cpu().numpy())
        target_texts = [self.reports[slide_id] for slide_id in slide_ids]
        self.print_results(slide_ids, pred_texts, target_texts)
        return slide_ids, pred_texts
