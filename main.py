import typer
import os
import torch
import random
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.datamodules.scout.dataloaders import EmbeddingDataLoader
from modules.loss import compute_nll_loss, compute_hybrid_loss
from modules.metrics.metrics import compute_scores
from modules.models.scout.report_gen_model import ReportGenModel
from modules.optimizers.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers.report_tokenizers import Tokenizer
from modules.trainers.trainer import Trainer
from utils.utils import get_params_for_key

app = typer.Typer(pretty_exceptions_enable=False)



def setup(devices):
    # Let torchrun set these; fallback for safety/debug
    torch.set_float32_matmul_precision('medium')
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Force loopback
    os.environ['MASTER_PORT'] = '30001'  # Or any free port >1024master_port

    if type(devices) == int:
        devices =','.join([str(i) for i in range(devices)])
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

    dist.init_process_group(backend='nccl')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def init(config_file_path, load_model=False):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    args = get_params_for_key(config_file_path, "train")
    args.lr *= world_size

    setup(args.devices)
    torch.cuda.set_device(local_rank)

    # tokenizer
    tokenizer = Tokenizer(args.reports_json_path, args.dataset_type)
    model = ReportGenModel(args, tokenizer).to(local_rank)
    if load_model:
        model = load_model_checkpoint(model, args)
    else:
        print(f"Rank {local_rank} has {sum(p.numel() for p in model.parameters())} parameters")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=args.unused_params)

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    criterion = compute_hybrid_loss if args.loss_type == 'hybrid' else compute_nll_loss
    metrics = compute_scores



    checkpoint_dir = args.save_dir
    if not os.path.exists(checkpoint_dir):
        if local_rank == 0:
            os.makedirs(checkpoint_dir)

    print('starting training')
    return model, criterion, metrics, optimizer, args, lr_scheduler, tokenizer, local_rank


@app.command()
def train(config_file_path: str='config.yaml', notes: str=''):
    model, criterion, metrics, optimizer, args, lr_scheduler, tokenizer,local_rank = init(config_file_path)

    # dataloader
    train_dataloader = EmbeddingDataLoader(args, tokenizer, split='train', shuffle=False)
    val_dataloader = EmbeddingDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = EmbeddingDataLoader(args, tokenizer, split='test', shuffle=False)

    trainer = Trainer(args, model, criterion, metrics, optimizer, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train(local_rank)
    trainer.test(local_rank)


@app.command()
def test(config_file_path: str='config.yaml', notes: str=''):
    model, criterion, metrics, optimizer, args, lr_scheduler, tokenizer, local_rank = init(config_file_path, load_model=True)

    test_dataloader = EmbeddingDataLoader(args, tokenizer, split='test', shuffle=False)
    trainer = Trainer(args, model, criterion, metrics, optimizer, lr_scheduler,
                      test_dataloader=test_dataloader)
    trainer.test(local_rank)

def load_model_checkpoint(model, args):
    resume_path = args.model_load_path
    state_dict = torch.load(resume_path)['state_dict']
    model.load_state_dict(state_dict)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)['state_dict']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint.items()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return  model

if __name__ == "__main__":
    app()
