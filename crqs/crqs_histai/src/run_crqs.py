# src/run_crqs.py

"""
HistAI CRQS pipeline.
"""

from pathlib import Path
from statistics import mean
from typing import Any

from crqs.crqs_common import CRQSPipeline, PipelineOptions
from crqs.crqs_histai.src.compute_metrics import compute_metrics
from crqs.crqs_histai.src.extract_fields import (
    extract_fields,
    learn_vocabulary,
    load_dataset_reports,
    load_vocabulary,
    save_vocabulary,
)


TARGET_KEYS = [
    "report",
    "target",
    "target_report",
    "ground_truth",
    "ground_truth_report",
]

PRED_KEYS = [
    "prediction",
    "predicted_report",
    "generated_report",
    "model_report",
    "pred",
]

METRIC_ORDER = [
    "id",
    "split",
    "cancer_type",
    "has_prediction",
    "CFC",
    "KIR",
    "HR",
    "CDS",
    "CRQS_raw",
    "CRQS_norm",
    "CFC_correct",
    "CFC_total",
    "KIR_correct",
    "KIR_total",
    "HR_unsupported",
    "HR_total",
    "CDS_discordant",
    "CDS_total",
]


class HistAICRQSPipeline(CRQSPipeline):
    dataset_name = "histai"

    def __init__(self, options: PipelineOptions):
        super().__init__(options)
        self.vocab_path = self.vocab_path or self.output_dir / "histai_vocab.json"

    def run(self) -> dict[str, Any]:
        self.ensure_output_dir()

        vocab = self.load_or_build_vocab()
        cases = self.flatten_dataset(self.load_json())
        extraction_rows, metric_rows = self.run_cases(cases, vocab)
        summary = self.summarize(cases, extraction_rows, metric_rows)

        self.write_outputs(extraction_rows, metric_rows, summary)
        self.print_summary(cases, extraction_rows, metric_rows, summary)
        return summary

    def load_or_build_vocab(self) -> dict[str, Any]:
        if self.options.rebuild_vocab or not self.vocab_path.exists():
            reports = load_dataset_reports(self.input_path)
            vocab = learn_vocabulary(reports, min_freq=self.options.min_count)
            save_vocabulary(vocab, self.vocab_path)

            print(f"Loaded {len(reports)} reports for vocabulary learning")
            print(f"Saved vocabulary to {self.vocab_path}")
            print(f"Learned {len(vocab.get('histology_terms', []))} histology terms")
            return vocab

        print(f"Loaded vocabulary from {self.vocab_path}")
        return load_vocabulary(self.vocab_path)

    def run_cases(
        self,
        cases: list[dict[str, Any]],
        vocab: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        extraction_rows = []
        metric_rows = []

        for idx, row in enumerate(cases):
            case_id = row.get("id", f"case_{idx}")
            split = row.get("_split", "all")
            cancer_type = row.get("cancer_type")

            target_text = self.first_present(row, TARGET_KEYS)
            prediction_text = self.first_present(row, PRED_KEYS)

            if not target_text:
                continue

            target_fields = extract_fields(target_text, vocab=vocab)
            pred_fields = None
            metrics = None

            if prediction_text:
                pred_fields = extract_fields(prediction_text, vocab=vocab)
                metrics = compute_metrics(target_fields, pred_fields)

            extraction_rows.append(
                {
                    "id": case_id,
                    "split": split,
                    "cancer_type": cancer_type,
                    "diagnosis_label": row.get("diagnosis"),
                    "target_text": target_text,
                    "prediction_text": prediction_text,
                    "target_fields": target_fields,
                    "prediction_fields": pred_fields,
                    "has_prediction": prediction_text is not None,
                }
            )

            if metrics is not None:
                metric_rows.append(
                    {
                        "id": case_id,
                        "split": split,
                        "cancer_type": cancer_type,
                        "has_prediction": True,
                        **metrics,
                    }
                )

        return extraction_rows, metric_rows

    def summarize(
        self,
        cases: list[dict[str, Any]],
        extraction_rows: list[dict[str, Any]],
        metric_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if metric_rows:
            summary = self.summarize_metric_rows(metric_rows)
            summary["mode"] = "crqs_evaluation"
        else:
            summary = self.summarize_extraction_rows(extraction_rows)
            summary["mode"] = "extraction_only"

        summary["input_path"] = str(self.input_path)
        summary["vocab_path"] = str(self.vocab_path)
        summary["n_cases_loaded"] = len(cases)
        summary["n_cases_processed"] = len(extraction_rows)
        summary["n_cases_with_predictions"] = sum(
            1 for row in extraction_rows if row.get("has_prediction")
        )
        return summary

    def write_outputs(
        self,
        extraction_rows: list[dict[str, Any]],
        metric_rows: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        self.write_json(self.output_dir / "histai_per_case_extractions.json", extraction_rows)
        self.write_csv(
            self.output_dir / "histai_per_case_metrics.csv",
            metric_rows,
            fieldnames=self.ordered_fieldnames(metric_rows, METRIC_ORDER) if metric_rows else None,
        )
        self.write_json(self.output_dir / "histai_summary.json", summary)
        self.write_csv(self.output_dir / "histai_summary.csv", [summary])

    def print_summary(
        self,
        cases: list[dict[str, Any]],
        extraction_rows: list[dict[str, Any]],
        metric_rows: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        print("\n" + "=" * 80)
        print("HistAI CRQS pipeline complete")
        print("=" * 80)
        print(f"Mode: {summary['mode']}")
        print(f"Cases loaded: {len(cases)}")
        print(f"Cases processed: {len(extraction_rows)}")
        print(f"Cases with predictions: {summary['n_cases_with_predictions']}")
        print(f"Per-case extractions JSON: {self.output_dir / 'histai_per_case_extractions.json'}")
        print(f"Per-case metrics CSV: {self.output_dir / 'histai_per_case_metrics.csv'}")
        print(f"Summary JSON: {self.output_dir / 'histai_summary.json'}")
        print(f"Summary CSV: {self.output_dir / 'histai_summary.csv'}")

        if metric_rows:
            print("\nSummary metrics:")
            for key, value in summary.items():
                if key.endswith("_mean") and value is not None:
                    print(f"  {key}: {value:.4f}")
        else:
            print("\nExtraction-only summary:")
            print(f"  Target reports: {summary.get('n_with_target_report')}")
            print(f"  Predictions: {summary.get('n_with_prediction')}")
            print("  Top extracted target fields:")
            for field, count in list(summary.get("target_field_presence_counts", {}).items())[:15]:
                print(f"    {field}: {count}")

    @staticmethod
    def flatten_dataset(data: Any) -> list[dict[str, Any]]:
        cases = []

        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    row = dict(row)
                    row["_split"] = row.get("_split", "all")
                    cases.append(row)

        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    for row in value:
                        if isinstance(row, dict):
                            row = dict(row)
                            row["_split"] = key
                            cases.append(row)
                elif isinstance(value, dict):
                    row = dict(value)
                    row["id"] = row.get("id", key)
                    row["_split"] = row.get("_split", "all")
                    cases.append(row)
        else:
            raise ValueError("Unsupported JSON structure. Expected list or dict.")

        return cases

    @staticmethod
    def first_present(row: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip() != "":
                return value
        return None

    @staticmethod
    def summarize_metric_rows(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
        metric_names = ["CFC", "KIR", "HR", "CDS", "CRQS_raw", "CRQS_norm"]
        summary = {"n_evaluated": len(metric_rows)}

        for metric in metric_names:
            values = [
                row[metric]
                for row in metric_rows
                if metric in row and row[metric] is not None
            ]
            summary[f"{metric}_mean"] = mean(values) if values else None

        return summary

    @staticmethod
    def summarize_extraction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {
            "n_cases": len(rows),
            "n_with_target_report": sum(1 for row in rows if row.get("target_text")),
            "n_with_prediction": sum(1 for row in rows if row.get("prediction_text")),
        }

        field_counter = {}
        for row in rows:
            target_fields = row.get("target_fields", {})
            for field in target_fields:
                field_counter[field] = field_counter.get(field, 0) + 1

        summary["target_field_presence_counts"] = dict(
            sorted(field_counter.items(), key=lambda x: x[1], reverse=True)
        )
        return summary


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    vocab_path: str | Path,
    min_freq: int = 2,
    rebuild_vocab: bool = False,
) -> dict[str, Any]:
    options = PipelineOptions(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        vocab_path=Path(vocab_path),
        min_count=min_freq,
        rebuild_vocab=rebuild_vocab,
    )
    return HistAICRQSPipeline(options).run()
