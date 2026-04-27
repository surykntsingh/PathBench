from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from utils.utils import read_json_file


class BaseReportDataset(Dataset, ABC):

    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_length = args.max_seq_length
        self.reports = self.load_reports(split, args.reports_json_path)
        self.slides = self.build_slides()

    def load_reports(self, split, reports_json_path):
        reports = read_json_file(reports_json_path)[split]
        return {report['id']: report['report'] for report in reports}

    def encode_report(self, slide_id):
        report_text = self.reports[slide_id]
        report_ids = self.tokenizer(report_text)
        seq_length = min(len(report_ids), self.max_seq_length)

        if len(report_ids) > self.max_seq_length:
            report_ids = report_ids[:self.max_seq_length]
        elif len(report_ids) < self.max_seq_length:
            report_ids.extend([0] * (self.max_seq_length - len(report_ids)))

        report_masks = [1] * seq_length + [0] * (self.max_seq_length - seq_length)
        return report_ids, report_masks

    def __len__(self):
        return len(self.slides)

    @abstractmethod
    def build_slides(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError


class BaseReportDataModule(pl.LightningDataModule, ABC):

    def __init__(self, args, tokenizer, shuffle=False):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_ds = self.build_dataset('train')
            self.val_ds = self.build_dataset('val')
        if stage in (None, 'test'):
            self.test_ds = self.build_dataset('test')

    @abstractmethod
    def build_dataset(self, split):
        raise NotImplementedError

    def train_dataloader(self):
        return self._build_dataloader(self.train_ds, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._build_dataloader(self.val_ds)

    def test_dataloader(self):
        return self._build_dataloader(self.test_ds)

    def _build_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        raise NotImplementedError

    @staticmethod
    def to_long_tensor(values):
        return torch.LongTensor(values)

    @staticmethod
    def to_float_tensor(values):
        return torch.FloatTensor(values)
