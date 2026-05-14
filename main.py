import typer
import os
import json
import torch
import random
import numpy as np

from modules.tokenizers.report_tokenizers import Tokenizer
import pytorch_lightning as pl
import torch.distributed as dist
from modules.trainers.trainer import Trainer
from utils.utils import get_params_for_key
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    report_model_cls, model = build_model(args, tokenizer)
    if load_model or args.resume:
        print(f'Loading model from {args.model_load_path}')
        report_model = report_model_cls.load_from_checkpoint(
            args.model_load_path,
            args=args,
            model=model,
            tokenizer=tokenizer,
        )
    else:
        report_model = report_model_cls(args, model, tokenizer)

    datamodule = build_datamodule(args, tokenizer)
    trainer = Trainer(args, tokenizer)

    return trainer, datamodule, report_model, tokenizer, args


def is_global_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def build_model(args, tokenizer):
    model_type = getattr(args, 'model_type', 'scout').lower()

    if model_type == 'scout':
        from modules.models.scout.scout_model import SCOUTModule
        from modules.models.scout_report_model import ScoutReportModel
        return ScoutReportModel, SCOUTModule(args, tokenizer)

    if model_type in {'wsi_caption', 'r2gen'}:
        from modules.models.wsi_caption.r2gen import R2GenModel
        from modules.models.caption_report_model import CaptionReportModel
        return CaptionReportModel, R2GenModel(args, tokenizer)

    if model_type == 'histgen':
        from modules.models.histgen.histgen_model import HistGenModel
        from modules.models.caption_report_model import CaptionReportModel
        return CaptionReportModel, HistGenModel(args, tokenizer)

    if model_type == 'bigen':
        from modules.models.bigen.r2gen import R2GenModel as BiGenModel
        from modules.models.bigen_report_model import BiGenReportModel
        return BiGenReportModel, BiGenModel(args, tokenizer)

    raise ValueError(f'Unsupported model_type: {model_type}')


def build_datamodule(args, tokenizer):
    model_type = getattr(args, 'model_type', 'scout').lower()

    if model_type == 'scout':
        from modules.datamodules.scout.base import ScoutDataModule
        return ScoutDataModule(args, tokenizer)

    if model_type in {'wsi_caption', 'r2gen'}:
        from modules.datamodules.wsi_caption.base import WSICaptionDataModule
        return WSICaptionDataModule(args, tokenizer)

    if model_type == 'histgen':
        from modules.datamodules.histgen.base import HistGenDataModule
        return HistGenDataModule(args, tokenizer)

    if model_type == 'bigen':
        from modules.datamodules.bigen.base import BiGenDataModule
        return BiGenDataModule(args, tokenizer)

    raise ValueError(f'Unsupported model_type: {model_type}')


@app.command("compute-metrics")
def compute_metrics(
    input_json_path: str = typer.Argument(..., help="JSON file with id, prediction, and ground-truth text."),
    output_json_path: Optional[str] = typer.Option(
        None,
        "--output-json-path",
        "-o",
        help="Path for the enriched per-sample metrics JSON.",
    ),
):
    from modules.metrics.metrics import compute_scores, compute_scores_per_sample

    input_path = Path(input_json_path)
    with input_path.open("r") as f:
        data = json.load(f)

    gts = {sample_id: [sample["target"]] for sample_id, sample in data.items()}
    preds = {sample_id: [sample["pred"]] for sample_id, sample in data.items()}
    per_sample_metrics = compute_scores_per_sample(gts, preds)
    overall_metrics = compute_scores(gts, preds)

    output_records = []
    for sample_id, sample in data.items():
        record = {
            "id": sample_id,
            "predicted": sample["pred"],
            "ground_truth": sample["target"],
            "metrics": per_sample_metrics[sample_id],
        }
        output_records.append(record)
        print(json.dumps(record, indent=2))

    output_path = Path(output_json_path) if output_json_path else input_path.with_name(f"{input_path.stem}_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output_records, f, indent=2)

    print("Overall metrics:")
    print(json.dumps(overall_metrics, indent=2))
    print(f"Saved per-sample metrics JSON to {output_path}")




@app.command()
def train(config_file_path: str='config.yaml', notes: str=''):

    trainer, datamodule, report_model,tokenizer, args = init(config_file_path)

    train_metrics, best_model_path = trainer.train(report_model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model training finished')
    print('Model testing begins')
    print(f'loading best model from {best_model_path}')
    report_model_cls, model = build_model(args, tokenizer)
    best_report_model = report_model_cls.load_from_checkpoint(
        best_model_path,
        args=args,
        model=model,
        tokenizer=tokenizer,
    )
    test_metrics = trainer.test(best_report_model, datamodule, fast_dev_run=args.fast_dev_run)
    if is_global_zero():
        print('model testing finished')
        metrics = {**train_metrics, **test_metrics, 'best_model_path': best_model_path}
        print(f'train_metrics: {train_metrics}, test_metrics: {test_metrics}')
        metrics['exp_notes'] = notes

        date = datetime.now()
        results_path = f'{args.output_dir}/metrics'
        os.makedirs(results_path, exist_ok=True)
        write_metrics(results_path, metrics, date)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@app.command()
def test(config_file_path: str='config.yaml'):
    trainer, datamodule, report_model, tokenizer, args = init(config_file_path, load_model=True)
    test_metrics = trainer.test(report_model, datamodule, fast_dev_run=args.fast_dev_run)
    if is_global_zero():
        print(f'test_metrics: {test_metrics}')
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def write_metrics(results_path, metrics, date):
    metrics['date'] = date.strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([metrics])


    metrics_df.to_csv(f'{results_path}/results.csv',mode='a')

if __name__ == "__main__":
    app()
