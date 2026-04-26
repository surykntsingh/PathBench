import torch
from torch.utils.data import Subset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from modules.datamodules.scout.datasets import EmbeddingDataset


class EmbeddingDataModule(pl.LightningDataModule):

    def __init__(self,args, tokenizer, shuffle = False):
        super().__init__()
        self.args = args
        self.test_ds = None
        self.val_ds = None
        self.train_ds = None
        self.shuffle = shuffle

        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_ds = EmbeddingDataset(self.args, self.tokenizer, 'train')
        self.val_ds = EmbeddingDataset(self.args, self.tokenizer, 'val')
        self.test_ds = EmbeddingDataset(self.args, self.tokenizer, 'test')


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=self.shuffle, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.args.batch_size, collate_fn = self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.args.batch_size, collate_fn = self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        slide_ids, slide_embedding, patch_embeddings, gecko_deep_embedding, gecko_concept_embedding, report_ids, report_masks = zip(*batch)
        report_ids = torch.LongTensor(report_ids)
        features = {
            'slide': pad_sequence(slide_embedding,batch_first=True),
            'patch': pad_sequence(patch_embeddings,batch_first=True),
            'gecko': {
                'deep': pad_sequence(gecko_deep_embedding,batch_first=True),
                'concept': pad_sequence(gecko_concept_embedding,batch_first=True)
            }
        }
        return slide_ids, features, report_ids, torch.FloatTensor(report_masks)
