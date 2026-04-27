import os

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence

from modules.datamodules.base import BaseReportDataModule, BaseReportDataset


class BiGenDataset(BaseReportDataset):

    def __init__(self, args, tokenizer, split):
        self.data_path_patch = args.data_path_patch
        self.data_path_kb = args.data_path_kb
        self.max_fea_length = args.max_fea_length
        super().__init__(args, tokenizer, split)

    def build_slides(self):
        files_patch = os.listdir(self.data_path_patch)
        slides = [
            slide_id for slide_id in self.reports.keys()
            if slide_id != 'TCGA-A2-A1G0-01Z-00-DX1.9ECB0B8A-EF4E-45A9-82AC-EF36375DEF65'
        ]

        print(f'dataset_type: {self.split} self.__slides: {len(slides)}')

        matched_slides = [
            '.'.join(file.split('.')[:-1])
            for file in files_patch
            if '.'.join(file.split('.')[:-1]) in slides
        ]
        print(f'dataset_type: {self.split}, files: {len(files_patch)}, slides: {len(matched_slides)}')
        return matched_slides

    def __getitem__(self, idx):
        slide_id = self.slides[idx]
        report_ids, report_masks = self.encode_report(slide_id)

        with h5py.File(f'{self.data_path_patch}/{slide_id}.h5', "r") as h5_file:
            patch_embedding = torch.tensor(h5_file["features"][:])[:self.max_fea_length]

        with h5py.File(f'{self.data_path_kb}/{slide_id}.h5', "r") as h5_file:
            kb_embedding = torch.tensor(h5_file["features"][:])[:self.max_fea_length]

        return slide_id, patch_embedding, kb_embedding, report_ids, report_masks


class BiGenDataModule(BaseReportDataModule):

    def build_dataset(self, split):
        return BiGenDataset(self.args, self.tokenizer, split)

    @staticmethod
    def collate_fn(batch):
        slide_ids, patch_embeddings, kb_embedding, report_ids, report_masks = zip(*batch)
        report_ids = BiGenDataModule.to_long_tensor(report_ids)
        features = {
            'patch': pad_sequence(patch_embeddings, batch_first=True),
            'kb': pad_sequence(kb_embedding, batch_first=True),
        }
        return slide_ids, features, report_ids, BiGenDataModule.to_float_tensor(report_masks)
