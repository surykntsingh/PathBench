"""
REG CRQS pipeline.
"""

import json
from pathlib import Path
from typing import Any

from crqs.crqs_common import CRQSPipeline, PipelineOptions
from crqs.crqs_reg.src.compute_metrics import compute_all_metrics
from crqs.crqs_reg.src.extract_fields import (
    extract_fields,
    learn_vocabulary_from_json,
    load_vocabulary,
    save_vocabulary,
)


class RegCRQSPipeline(CRQSPipeline):
    dataset_name = "reg"

    def __init__(self, options: PipelineOptions):
        super().__init__(options)
        self.vocab_path = self.vocab_path or self.output_dir / "reg_vocabulary.json"

    def run(self) -> dict[str, Any]:
        self.ensure_output_dir()

        vocab = self.build_vocab()
        records = self.load_json()
        case_rows, detailed_json = self.run_cases(records, vocab)
        summary = self.summarize(case_rows)

        self.write_outputs(case_rows, detailed_json, summary)
        self.print_summary(summary)
        return summary

    def build_vocab(self) -> dict[str, Any]:
        print("Learning vocabulary from input file...")
        vocab = learn_vocabulary_from_json(self.input_path, min_count=self.options.min_count)
        save_vocabulary(vocab, self.vocab_path)
        vocab = load_vocabulary(self.vocab_path)

        print(f"Saved vocabulary to {self.vocab_path}")
        print(f"Learned {len(vocab['diagnosis_terms'])} diagnosis terms.")
        return vocab

    def run_cases(
        self,
        records: dict[str, dict[str, str]],
        vocab: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        case_rows = []
        detailed_json = {}

        print("Evaluating REG")
        for case_id, item in records.items():
            target_report = item.get("target", "")
            pred_report = item.get("pred", "")

            target_facts = extract_fields(target_report, vocab=vocab)
            pred_facts = extract_fields(pred_report, vocab=vocab)
            metrics = compute_all_metrics(target_facts, pred_facts)

            raw_crqs = metrics["CRQS"]
            norm_crqs = self.normalize_crqs(raw_crqs)

            case_rows.append(
                {
                    "case_id": case_id,
                    "CFC": metrics["CFC"],
                    "KIR": metrics["KIR"],
                    "HR": metrics["HR"],
                    "CDS": metrics["CDS"],
                    "CRQS_raw": raw_crqs,
                    "CRQS_norm": norm_crqs,
                    "target_report": target_report,
                    "pred_report": pred_report,
                    "target_facts": json.dumps(target_facts),
                    "pred_facts": json.dumps(pred_facts),
                    "CFC_correct": json.dumps(metrics["details"]["CFC"]["correct"]),
                    "CFC_missed": json.dumps(metrics["details"]["CFC"]["missed"]),
                    "KIR_correct": json.dumps(metrics["details"]["KIR"]["correct"]),
                    "KIR_missed": json.dumps(metrics["details"]["KIR"]["missed"]),
                    "HR_hallucinated": json.dumps(metrics["details"]["HR"]["hallucinated"]),
                    "CDS_discordant": json.dumps(metrics["details"]["CDS"]["discordant"]),
                }
            )

            detailed_json[case_id] = {
                "target_report": target_report,
                "pred_report": pred_report,
                "target_facts": target_facts,
                "pred_facts": pred_facts,
                "metrics": {
                    "CFC": metrics["CFC"],
                    "KIR": metrics["KIR"],
                    "HR": metrics["HR"],
                    "CDS": metrics["CDS"],
                    "CRQS_raw": raw_crqs,
                    "CRQS_norm": norm_crqs,
                },
                "details": metrics["details"],
            }

        return case_rows, detailed_json

    def summarize(self, case_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "num_cases": len(case_rows),
            "CFC": self.mean_value(case_rows, "CFC"),
            "KIR": self.mean_value(case_rows, "KIR"),
            "HR": self.mean_value(case_rows, "HR"),
            "CDS": self.mean_value(case_rows, "CDS"),
            "CRQS_raw": self.mean_value(case_rows, "CRQS_raw"),
            "CRQS_norm": self.mean_value(case_rows, "CRQS_norm"),
        }

    def write_outputs(
        self,
        case_rows: list[dict[str, Any]],
        detailed_json: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        case_csv_path = self.output_dir / "reg_case_results.csv"
        case_json_path = self.output_dir / "reg_case_results.json"
        summary_csv_path = self.output_dir / "reg_summary.csv"
        summary_json_path = self.output_dir / "reg_summary.json"

        self.write_csv(case_csv_path, case_rows, fieldnames=list(case_rows[0].keys()) if case_rows else None)
        self.write_json(case_json_path, detailed_json)
        self.write_csv(summary_csv_path, [summary], fieldnames=list(summary.keys()))
        self.write_json(summary_json_path, summary)

        self.output_paths = [
            case_csv_path,
            case_json_path,
            summary_csv_path,
            summary_json_path,
        ]

    def print_summary(self, summary: dict[str, Any]) -> None:
        print("\nSummary:")
        for key, value in summary.items():
            if key == "num_cases":
                print(f"{key}: {value}")
            elif value is not None:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: None")

        print("\nSaved outputs:")
        for path in self.output_paths:
            print(path)

    @staticmethod
    def normalize_crqs(raw_crqs: float) -> float:
        return raw_crqs / 0.7


def run_crqs(input_path: Path, output_dir: Path, min_vocab_count: int = 1) -> dict[str, Any]:
    options = PipelineOptions(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        min_count=min_vocab_count,
    )
    return RegCRQSPipeline(options).run()
