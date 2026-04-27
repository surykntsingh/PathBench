import os

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence

from modules.datamodules.base import BaseReportDataModule, BaseReportDataset


class ScoutDataset(BaseReportDataset):

    def __init__(self, args, tokenizer, split):
        self.data_path_slide = args.data_path_slide
        self.data_path_patch = args.data_path_patch
        self.data_path_concept = args.data_path_concept
        super().__init__(args, tokenizer, split)

    def build_slides(self):
        files_slide = os.listdir(self.data_path_slide)
        files_patch = os.listdir(self.data_path_patch)
        files_concept = os.listdir(self.data_path_concept)
        slides = [
            slide_id for slide_id in self.reports.keys()
            if slide_id != 'TCGA-A2-A1G0-01Z-00-DX1.9ECB0B8A-EF4E-45A9-82AC-EF36375DEF65'
        ]

        print(f'dataset_type: {self.split} self.__slides: {len(slides)}')

        matched_slides = [
            '.'.join(file.split('.')[:-1])
            for file in files_slide
            if file in files_patch
            and f'{file[:12]}.h5' in files_concept
            and '.'.join(file.split('.')[:-1]) in slides
        ]

        print(
            f'dataset_type: {self.split}, files: {len(files_slide)}, '
            f'files_1: {len(files_patch)}, files_2: {len(files_concept)} slides: {len(matched_slides)}'
        )
        return matched_slides

    def __getitem__(self, idx):
        slide_id = self.slides[idx]
        report_ids, report_masks = self.encode_report(slide_id)

        with h5py.File(f'{self.data_path_slide}/{slide_id}.h5', "r") as h5_file:
            slide_embedding = torch.tensor(h5_file["features"][:])

        with h5py.File(f'{self.data_path_patch}/{slide_id}.h5', "r") as h5_file:
            patch_embedding = torch.tensor(h5_file["features"][:])

        with h5py.File(f'{self.data_path_concept}/{slide_id[:12]}.h5', "r") as h5_file:
            gecko_deep_embedding = torch.tensor(h5_file["bag_feats_deep"][:])
            gecko_concept_embedding = torch.tensor(h5_file["bag_feats"][:])

        return (
            slide_id,
            slide_embedding,
            patch_embedding,
            gecko_deep_embedding,
            gecko_concept_embedding,
            report_ids,
            report_masks,
        )


class ScoutDataModule(BaseReportDataModule):

    def build_dataset(self, split):
        return ScoutDataset(self.args, self.tokenizer, split)

    @staticmethod
    def collate_fn(batch):
        slide_ids, slide_embedding, patch_embeddings, gecko_deep_embedding, gecko_concept_embedding, report_ids, report_masks = zip(*batch)
        report_ids = ScoutDataModule.to_long_tensor(report_ids)
        features = {
            'slide': pad_sequence(slide_embedding, batch_first=True),
            'patch': pad_sequence(patch_embeddings, batch_first=True),
            'gecko': {
                'deep': pad_sequence(gecko_deep_embedding, batch_first=True),
                'concept': pad_sequence(gecko_concept_embedding, batch_first=True),
            },
        }
        return slide_ids, features, report_ids, ScoutDataModule.to_float_tensor(report_masks)
