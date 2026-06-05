# PathBench

PathBench is a PyTorch Lightning codebase for pathology report generation from
whole-slide image (WSI) embeddings. It trains and evaluates captioning/report
generation models against pathology report text using COCO-style language
metrics such as BLEU, METEOR, and ROUGE-L.
It also includes CRQS evaluation pipelines for clinical fact-based scoring of
generated pathology reports.

The repository currently includes implementations and configuration examples
for:

- `scout`: report generation from slide-level, patch-level, and concept
  embeddings.
- `wsi_caption` / `r2gen`: WSI captioning from patch embeddings.
- `histgen`: HistGen-style report generation from patch embeddings.
- `bigen`: report generation from patch embeddings plus a second knowledge-base
  embedding stream.

## Repository Layout

```text
.
├── main.py                         # Typer CLI for train/test runs
├── crqs/                           # CRQS scoring pipelines and dataset configs
├── *_config.yaml                   # Example training configs
├── modules/
│   ├── datamodules/                # Dataset and dataloader definitions
│   ├── models/                     # Model wrappers and architectures
│   ├── metrics/                    # BLEU/METEOR/ROUGE evaluation code
│   ├── optimizers/
│   ├── tokenizers/
│   └── trainers/
└── utils/
```

## Requirements

Use Python 3.10 or newer with a CUDA-enabled PyTorch install for full training.
The code uses GPU acceleration through PyTorch Lightning.

Core Python packages:

```bash
python3 -m pip install torch pytorch-lightning typer pyyaml h5py numpy pandas scikit-learn tqdm einops
```

Notes:

- Install the PyTorch build that matches your CUDA version from the official
  PyTorch instructions.
- METEOR evaluation in `modules/metrics/pycocoevalcap` may require Java to be
  available on `PATH`.
- This repository does not currently include a pinned `requirements.txt`, so
  package versions should be matched to the environment used for your
  experiments.

## Data Format

Training is driven by a YAML config file. Each config points to a report split
JSON file and one or more directories of precomputed WSI features.

The report JSON must contain `train`, `val`, and `test` keys. Each split is a
list of examples with this shape:

```json
{
  "train": [
    {
      "id": "SLIDE_ID",
      "report": "Ground-truth pathology report text."
    }
  ],
  "val": [],
  "test": []
}
```

Feature files are expected to be HDF5 files named `<SLIDE_ID>.h5`. The required
datasets inside each HDF5 file depend on the selected model:

| `model_type` | Required config paths | Expected HDF5 datasets |
| --- | --- | --- |
| `wsi_caption` / `r2gen` | `data_path_patch` | `features` |
| `histgen` | `data_path_patch` | `features` |
| `bigen` | `data_path_patch`, `data_path_kb` | `features` in both directories |
| `scout` | `data_path_slide`, `data_path_patch`, `data_path_concept` | `features` for slide/patch files, and `bag_feats_deep` plus `bag_feats` for concept files |

The slide ids in the report JSON must match the HDF5 filenames without the
`.h5` suffix.

## Configure a Run

Start from one of the provided configs:

- `wsi_caption_reg_config.yaml`
- `histgen_reg_config.yaml`
- `bigen_reg_config.yaml`
- `scout_reg_config.yaml`
- `scout_tcga_brca_config.yaml`

At minimum, update these fields for your environment:

- `model_type`: one of `scout`, `wsi_caption`, `r2gen`, `histgen`, or `bigen`.
- `reports_json_path`: path to the split JSON file.
- Feature paths such as `data_path_patch`, `data_path_slide`,
  `data_path_concept`, and `data_path_kb`.
- `output_dir`: directory for checkpoints, metrics, and predictions.
- `devices`: GPU count or comma-separated GPU ids, for example `1` or `0,1`.
- `fast_dev_run`: set to `true` for a quick smoke test.
- `model_load_path`: checkpoint path used by `test` or when `resume: true`.

## Train

Run training with a config file:

```bash
python3 main.py train --config-file-path wsi_caption_config.yaml
```

Optionally attach a note to the metrics row:

```bash
python3 main.py train --config-file-path scout_config.yaml --notes "SCOUT REG baseline"
```

Training writes checkpoints under:

```text
<output_dir>/ckpt/
```

After training, the CLI loads the best checkpoint and runs test evaluation. It
then appends metrics to:

```text
<output_dir>/metrics/results.csv
```

## Test an Existing Checkpoint

Set `model_load_path` in the config to the checkpoint you want to evaluate, then
run:

```bash
python3 main.py test --config-file-path wsi_caption_config.yaml
```

Test predictions are written to:

```text
<output_dir>/predictions.json
```

## Calculate CRQS Scores

CRQS scores can be calculated for generated pathology reports with the unified
entrypoint in `crqs/run_crqs.py`. Run it from the repository root so `crqs` is
available as a package.

```bash
python3 -m crqs.run_crqs \
  --dataset reg \
  --config crqs/crqs_reg/src/config.py \
  --input /path/to/predictions.json \
  --output-dir /path/to/crqs_outputs
```

The input JSON should be keyed by case id. Each case needs a ground-truth report
in `target` and a generated report in `pred`:

```json
{
  "CASE_ID": {
    "target": "Ground-truth pathology report text.",
    "pred": "Generated pathology report text."
  }
}
```

Supported datasets are `reg`, `histai`, and `tcga`:

```bash
python3 -m crqs.run_crqs --dataset reg --config crqs/crqs_reg/src/config.py --input predictions_reg.json --output-dir outputs/crqs_reg
python3 -m crqs.run_crqs --dataset histai --config crqs/crqs_histai/src/config.py --input predictions_histai.json --output-dir outputs/crqs_histai
python3 -m crqs.run_crqs --dataset tcga --config crqs/crqs_tcga/src/config.py --input predictions_tcga.json --output-dir outputs/crqs_tcga
```

Useful options:

- `--vocab`: reuse a vocabulary JSON for HistAI or TCGA.
- `--rebuild-vocab`: force vocabulary rebuilding.
- `--min-count`: minimum term frequency for vocabulary learning.
- `--limit`: process only the first N TCGA cases for debugging.

CRQS outputs include per-case metrics, extracted clinical facts, summary JSON,
and summary CSV files under the selected `--output-dir`.

## Quick Smoke Test

For a short validation pass before launching a full experiment:

1. Copy one of the config files.
2. Point it to a small report split and matching feature directory.
3. Set `fast_dev_run: true`.
4. Set `devices: 1`.
5. Run:

```bash
python3 main.py train --config-file-path your_config.yaml
```

## Outputs

PathBench produces:

- Lightning checkpoints in `<output_dir>/ckpt/`.
- Test predictions in `<output_dir>/predictions.json`.
- A metrics CSV in `<output_dir>/metrics/results.csv`.
- CRQS per-case and summary files in the CRQS `--output-dir`.
- Console logs with train, validation, and test metrics.

## License

This project is released under the terms in [LICENSE](LICENSE).
