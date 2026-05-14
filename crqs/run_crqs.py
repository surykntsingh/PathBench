"""
Single entry point for CRQS evaluation across supported datasets.

Examples:
    python -m crqs.run_crqs --dataset reg --config crqs/crqs_reg/src/config.py --input crqs/crqs_reg/data/predictions_reg.json --output-dir crqs/crqs_reg/outputs/reg_run
    python -m crqs.run_crqs --dataset histai --config crqs/crqs_histai/src/config.py --input crqs/crqs_histai/data/histai_hoptimus.json --output-dir crqs/crqs_histai/outputs/histai_hoptimus
    python -m crqs.run_crqs --dataset tcga --config crqs/crqs_tcga/src/config.py --input crqs/crqs_tcga/data/tcga_uni.json --output-dir crqs/crqs_tcga/outputs/tcga_uni
"""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

from crqs.crqs_common import PipelineOptions


DATASETS = ("reg", "histai", "tcga")

CONFIG_MODULES = {
    "reg": "crqs.crqs_reg.src.config",
    "histai": "crqs.crqs_histai.src.config",
    "tcga": "crqs.crqs_tcga.src.config",
}

PIPELINE_CLASSES = {
    "reg": ("crqs.crqs_reg.src.run_crqs", "RegCRQSPipeline"),
    "histai": ("crqs.crqs_histai.src.run_crqs", "HistAICRQSPipeline"),
    "tcga": ("crqs.crqs_tcga.src.run_crqs", "TCGACRQSPipeline"),
}

DEFAULT_MIN_COUNTS = {
    "reg": 1,
    "histai": 2,
    "tcga": 2,
}


def resolve_cli_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_config_module(dataset: str, config_path: Path) -> None:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    module_name = CONFIG_MODULES[dataset]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def resolve_vocab_path(dataset: str, vocab_path: str | None, output_dir: Path) -> Path | None:
    if vocab_path:
        return resolve_cli_path(vocab_path)

    if dataset == "histai":
        return output_dir / "histai_vocab.json"

    if dataset == "tcga":
        return output_dir / "tcga_vocab.json"

    return None


def build_pipeline(dataset: str, options: PipelineOptions):
    module_name, class_name = PIPELINE_CLASSES[dataset]
    module = importlib.import_module(module_name)
    pipeline_cls = getattr(module, class_name)
    return pipeline_cls(options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CRQS evaluation for REG, HistAI, or TCGA datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="Dataset pipeline to run.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Prediction JSON path.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Dataset config Python file path.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory for CRQS outputs.",
    )
    parser.add_argument(
        "--vocab",
        "--vocab-path",
        "--vocab_path",
        dest="vocab_path",
        default=None,
        help="Vocabulary JSON path for HistAI/TCGA.",
    )
    parser.add_argument(
        "--min-count",
        "--min_count",
        dest="min_count",
        type=int,
        default=None,
        help="Minimum frequency for learned vocabulary terms.",
    )
    parser.add_argument(
        "--rebuild-vocab",
        "--force-relearn-vocab",
        dest="rebuild_vocab",
        action="store_true",
        help="Force vocabulary rebuild for HistAI/TCGA.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional TCGA case limit for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset

    input_path = resolve_cli_path(args.input)
    config_path = resolve_cli_path(args.config)
    output_dir = resolve_cli_path(args.output_dir)
    vocab_path = resolve_vocab_path(dataset, args.vocab_path, output_dir)
    min_count = args.min_count if args.min_count is not None else DEFAULT_MIN_COUNTS[dataset]

    if args.limit is not None and dataset != "tcga":
        raise ValueError("--limit is currently supported only by the TCGA pipeline.")

    print(f"Dataset: {dataset}")
    print(f"Input: {input_path}")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    if vocab_path:
        print(f"Vocabulary: {vocab_path}")

    load_config_module(dataset, config_path)

    options = PipelineOptions(
        input_path=input_path,
        output_dir=output_dir,
        vocab_path=vocab_path,
        min_count=min_count,
        rebuild_vocab=args.rebuild_vocab,
        limit=args.limit,
    )
    build_pipeline(dataset, options).run()


if __name__ == "__main__":
    main()
