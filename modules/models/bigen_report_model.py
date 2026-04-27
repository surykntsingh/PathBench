from modules.models.caption_report_model import CaptionReportModel


class BiGenReportModel(CaptionReportModel):

    def get_model_inputs(self, features):
        return features['patch'], features['kb']
