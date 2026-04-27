import typer
import os
import torch
import random
import numpy as np

from modules.datamodules.scout.datamodules import EmbeddingDataModule
from modules.models.ReportModel import ReportModel
from modules.models.scout.scout_model import SCOUTModule
from modules.tokenizers.report_tokenizers import Tokenizer
import pytorch_lightning as pl
from modules.trainers.trainer import Trainer
from utils.utils import get_params_for_key
import pandas as pd
from datetime import datetime

app = typer.Typer(pretty_exceptions_enable=False)


def init_seeds(seed=42, cuda_deterministic=False):
    pl.seed_everything(seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    # torch.autograd.set_detect_anomaly(True)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def init(config_file_path, load_model=False):
    init_seeds(0)
    args = get_params_for_key(config_file_path, "train")
    tokenizer = Tokenizer(args.reports_json_path, args.dataset_type)
    model = SCOUTModule(args, tokenizer)
    if load_model or args.resume:
        print(f'Loading model from {args.model_load_path}')
        report_model = ReportModel.load_from_checkpoint(args.model_load_path, args=args,model=model, tokenizer=tokenizer)
    else:
        report_model = ReportModel(args, model, tokenizer)

    datamodule = EmbeddingDataModule(args, tokenizer)
    trainer = Trainer(args, tokenizer)

    return trainer, datamodule, report_model, tokenizer, args




@app.command()
def train(config_file_path: str='config.yaml', notes: str=''):

    trainer, datamodule, report_model,tokenizer, args = init(config_file_path)

    train_metrics, best_model_path = trainer.train(report_model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model training finished')
    print('Model testing begins')
    print(f'loading best model from {best_model_path}')
    model = SCOUTModule(args, tokenizer)
    best_report_model = ReportModel.load_from_checkpoint(best_model_path, args=args, model=model, tokenizer=tokenizer)
    test_metrics = trainer.test(best_report_model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model testing finished')
    metrics = {**train_metrics, **test_metrics, 'best_model_path': best_model_path}
    print(f'train_metrics: {train_metrics}, test_metrics: {test_metrics}')
    metrics['exp_notes'] = notes


    date = datetime.now()
    args.ckpt_path += f'/{date.strftime("%Y%m%d")}/{date.strftime("%H%M%S")}'

    results_path = f'{args.output_dir}/metrics'
    os.makedirs(results_path, exist_ok=True)
    write_metrics(results_path, metrics, date)


@app.command()
def test(config_file_path: str='config.yaml'):
    trainer, datamodule, report_model, tokenizer, args = init(config_file_path, load_model=True)
    test_metrics = trainer.test(report_model, datamodule, fast_dev_run=args.fast_dev_run)
    print(f'test_metrics: {test_metrics}')

def write_metrics(results_path, metrics, date):
    metrics['date'] = date.strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([metrics])


    metrics_df.to_csv(f'{results_path}/results.csv',mode='a')

if __name__ == "__main__":
    app()
