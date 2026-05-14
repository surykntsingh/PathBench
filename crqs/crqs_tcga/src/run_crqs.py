# src/run_crqs.py

import json
from pathlib import Path
from typing import Any

from crqs.crqs_common import CRQSPipeline, PipelineOptions
from crqs.crqs_tcga.src.compute_metrics import compute_crqs
from crqs.crqs_tcga.src.extract_fields import extract_fields, learn_vocabulary



METRIC_COLUMNS = [
    "CFC",
    "KIR",
    "HR",
    "CDS",
    "CRQS_raw",
    "CRQS_norm",
    "n_target_fields",
    "n_pred_fields",
    "n_target_key_fields",
]


class TCGACRQSPipeline(CRQSPipeline):
    dataset_name = "tcga"

    def __init__(self, options: PipelineOptions):
        super().__init__(options)
        self.vocab_path = self.vocab_path or self.output_dir / "tcga_vocab.json"

    def run(self) -> dict[str, Any]:
        self.ensure_output_dir()

        pred_data = self.load_json()
        self.validate_input(pred_data)

        vocab = self.load_or_learn_vocab(pred_data)
        per_case = self.run_cases(pred_data, vocab)
        summary = self.summarize(per_case, pred_data)

        self.write_outputs(vocab, per_case, summary)
        self.print_summary(summary)
        return summary

    def load_or_learn_vocab(self, pred_data: dict[str, dict[str, str]]) -> dict[str, Any]:
        if self.vocab_path.exists() and not self.options.rebuild_vocab:
            return self.load_json(self.vocab_path)

        vocab_dataset = self.prediction_dict_to_vocab_dataset(pred_data)
        vocab = learn_vocabulary(vocab_dataset, min_count=self.options.min_count)
        self.write_json(self.vocab_path, vocab)
        return vocab

    def run_cases(
        self,
        pred_data: dict[str, dict[str, str]],
        vocab: dict[str, Any],
    ) -> list[dict[str, Any]]:
        items = list(pred_data.items())
        if self.options.limit is not None:
            items = items[: self.options.limit]

        per_case = []
        for i, (case_id, item) in enumerate(items, 1):
            per_case.append(
                self.run_case(
                    case_id=case_id,
                    target_report=item.get("target", ""),
                    pred_report=item.get("pred", ""),
                    vocab=vocab,
                )
            )

            if i % 50 == 0:
                print(f"Processed {i}/{len(items)} cases")

        return per_case

    def run_case(
        self,
        case_id: str,
        target_report: str,
        pred_report: str,
        vocab: dict[str, Any],
    ) -> dict[str, Any]:
        target_fields = extract_fields(target_report, vocab)
        pred_fields = extract_fields(pred_report, vocab)
        metrics = compute_crqs(target_fields, pred_fields)

        return {
            "case_id": case_id,
            **metrics,
            "target_fields": target_fields,
            "pred_fields": pred_fields,
            "target_report": target_report,
            "pred_report": pred_report,
        }

    def summarize(
        self,
        per_case: list[dict[str, Any]],
        pred_data: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        summary = {
            "n_cases": len(per_case),
            "main_scores": {
                "CRQS_raw": self.mean_value(per_case, "CRQS_raw", ndigits=4),
                "CRQS_norm": self.mean_value(per_case, "CRQS_norm", ndigits=4),
            },
            "components": {
                "CFC": self.mean_value(per_case, "CFC", ndigits=4),
                "KIR": self.mean_value(per_case, "KIR", ndigits=4),
                "HR": self.mean_value(per_case, "HR", ndigits=4),
                "CDS": self.mean_value(per_case, "CDS", ndigits=4),
            },
            "field_counts": {
                "target_fields": self.mean_value(per_case, "n_target_fields", ndigits=4),
                "pred_fields": self.mean_value(per_case, "n_pred_fields", ndigits=4),
                "target_key_fields": self.mean_value(per_case, "n_target_key_fields", ndigits=4),
            },
            "mode": "crqs_evaluation",
            "input_path": str(self.input_path),
            "vocab_path": str(self.vocab_path),
            "n_cases_loaded": len(pred_data),
            "n_cases_processed": len(per_case),
            "n_cases_with_predictions": sum(1 for item in pred_data.values() if item.get("pred")),
        }
        return summary

    def write_outputs(
        self,
        vocab: dict[str, Any],
        per_case: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        self.write_json(self.output_dir / "tcga_vocab.json", vocab)
        self.write_json(self.output_dir / "per_case_metrics.json", per_case)
        self.write_json(self.output_dir / "summary_metrics.json", summary)

        csv_rows = [self.flatten_case_for_csv(row) for row in per_case]
        self.write_csv(self.output_dir / "per_case_metrics.csv", csv_rows, fieldnames=list(csv_rows[0].keys()) if csv_rows else None)
        self.write_csv(self.output_dir / "summary_metrics.csv", [summary])

    def print_summary(self, summary: dict[str, Any]) -> None:
        print("\nCRQS run complete")
        print(f"Input: {self.input_path}")
        print(f"Output directory: {self.output_dir}")
        print(json.dumps(summary, indent=2))

    @staticmethod
    def prediction_dict_to_vocab_dataset(
        pred_data: dict[str, dict[str, str]],
    ) -> dict[str, list[dict[str, str]]]:
        rows = []

        for case_id, item in pred_data.items():
            target = item.get("target", "")
            pred = item.get("pred", "")

            if target:
                rows.append({"id": case_id, "report": target})
            if pred:
                rows.append({"id": case_id + "_pred", "report": pred})

        return {"all": rows}

    @staticmethod
    def flatten_case_for_csv(row: dict[str, Any]) -> dict[str, Any]:
        flat = {"case_id": row["case_id"]}

        for metric in METRIC_COLUMNS:
            flat[metric] = row.get(metric)

        flat["target_fields_json"] = json.dumps(row.get("target_fields", {}), ensure_ascii=False)
        flat["pred_fields_json"] = json.dumps(row.get("pred_fields", {}), ensure_ascii=False)
        return flat

    @staticmethod
    def validate_input(pred_data: Any) -> None:
        if not isinstance(pred_data, dict):
            raise ValueError("Input JSON must be a dictionary keyed by case ID.")

        bad = []
        for case_id, item in pred_data.items():
            if not isinstance(item, dict):
                bad.append(case_id)
                continue
            if "target" not in item or "pred" not in item:
                bad.append(case_id)

        if bad:
            raise ValueError(
                f"Found {len(bad)} malformed cases. Each case must contain 'target' and 'pred'. "
                f"Example bad case IDs: {bad[:5]}"
            )


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    vocab_path: str | Path | None = None,
    min_count: int = 2,
    force_relearn_vocab: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    options = PipelineOptions(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        vocab_path=Path(vocab_path) if vocab_path else None,
        min_count=min_count,
        rebuild_vocab=force_relearn_vocab,
        limit=limit,
    )
    return TCGACRQSPipeline(options).run()
