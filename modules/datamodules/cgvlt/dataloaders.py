import math
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from modules.datamodules.cgvlt.datasets import EmbeddingDataset


class EmbeddingDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        # self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        self.dataset = EmbeddingDataset(args, tokenizer, split)

        if split == 'train':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = SequentialDistributedSampler(self.dataset ,self.args.batch_size)


        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'sampler': self.sampler
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(batch, device='cuda'):
        slide_ids, slide_embedding, patch_embeddings, gecko_deep_embedding, gecko_concept_embedding, report_ids, report_masks = zip(
            *batch)
        report_ids = torch.LongTensor(report_ids)#.to(device)
        features = {
            'slide': pad_sequence(slide_embedding, batch_first=True), #.to(device),
            'patch': pad_sequence(patch_embeddings, batch_first=True), #.to(device),
            'gecko': {
                'deep': pad_sequence(gecko_deep_embedding, batch_first=True), #.to(device),
                'concept': pad_sequence(gecko_concept_embedding, batch_first=True), #.to(device)
            }
        }
        return slide_ids, features, report_ids, torch.FloatTensor(report_masks)


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indices sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int \
            (math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples