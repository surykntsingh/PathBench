import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class PipelineOptions:
    input_path: Path
    output_dir: Path
    vocab_path: Path | None = None
    min_count: int = 2
    rebuild_vocab: bool = False
    limit: int | None = None


class CRQSPipeline:
    dataset_name = "crqs"

    def __init__(self, options: PipelineOptions):
        self.options = options
        self.input_path = Path(options.input_path)
        self.output_dir = Path(options.output_dir)
        self.vocab_path = Path(options.vocab_path) if options.vocab_path else None

    def run(self) -> dict[str, Any]:
        raise NotImplementedError

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def load_json(self, path: str | Path | None = None) -> Any:
        path = Path(path) if path else self.input_path
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, path: str | Path, obj: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def write_csv(
        self,
        path: str | Path,
        rows: list[dict[str, Any]],
        fieldnames: list[str] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return

        if fieldnames is None:
            fieldnames = sorted({key for row in rows for key in row.keys()})

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def mean_value(rows: list[dict[str, Any]], key: str, ndigits: int | None = None) -> float | None:
        values = [
            row[key]
            for row in rows
            if key in row and isinstance(row[key], (int, float))
        ]
        if not values:
            return None

        value = mean(values)
        return round(value, ndigits) if ndigits is not None else value

    @staticmethod
    def ordered_fieldnames(
        rows: list[dict[str, Any]],
        preferred_order: list[str],
    ) -> list[str]:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        ordered = [key for key in preferred_order if key in fieldnames]
        ordered.extend(key for key in fieldnames if key not in ordered)
        return ordered
