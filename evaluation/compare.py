"""Lightweight cross-variant comparison report.

Reads one ``metrics.json`` per variant directory and emits deterministic
``comparison.json`` and ``comparison.md``. These are file-localization
metrics; they are never labelled "RCA Accuracy".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"

METRIC_ROWS = [
    ("top_1_file_accuracy", "Top-1 File Accuracy"),
    ("any_file_recall_at_3", "Any File Recall@3"),
    ("any_file_recall_at_5", "Any File Recall@5"),
    ("all_file_recall_at_3", "All File Recall@3"),
    ("all_file_recall_at_5", "All File Recall@5"),
    ("mean_gold_file_recall_at_5", "Mean Gold File Recall@5"),
    ("coverage", "Coverage"),
    ("invalid_output_rate", "Invalid-Output Rate"),
    ("latency_p50_seconds", "Latency P50 (s)"),
    ("latency_p95_seconds", "Latency P95 (s)"),
    ("mean_llm_calls_per_case", "LLM Calls/Case"),
    ("mean_total_tokens_per_case", "Tokens/Case"),
    ("mean_reasoning_tokens_per_case", "Reasoning Tokens/Case"),
]


def load_variant_metrics(variant_dir: str | Path) -> dict:
    """Load one variant directory's aggregate metrics."""
    directory = Path(variant_dir)
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"no metrics.json in variant dir: {directory}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "variant": directory.name,
        "config": data.get("config", {}),
        "aggregate": data.get("aggregate", {}),
    }


def build_comparison(variant_metrics: list[dict]) -> dict:
    """Build the deterministic comparison document."""
    variants = sorted(variant_metrics, key=lambda entry: entry["variant"])
    rows = []
    for key, label in METRIC_ROWS:
        rows.append(
            {
                "metric": key,
                "label": label,
                "values": {
                    entry["variant"]: entry["aggregate"].get(key)
                    for entry in variants
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "note": "File-localization metrics only; not RCA Accuracy.",
        "variants": [
            {"variant": entry["variant"], "config": entry["config"]}
            for entry in variants
        ],
        "rows": rows,
    }


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(comparison: dict) -> str:
    """Render the comparison table as deterministic markdown."""
    variants = [entry["variant"] for entry in comparison["variants"]]
    lines = [
        "# RootTrace Benchmark Ablation Comparison",
        "",
        "File-localization metrics only; not RCA Accuracy.",
        "",
        "| metric | " + " | ".join(variants) + " |",
        "| --- |" + " --- |" * len(variants),
    ]
    for row in comparison["rows"]:
        values = [row["values"].get(variant) for variant in variants]
        lines.append(
            "| {label} | {values} |".format(
                label=row["label"],
                values=" | ".join(_format(value) for value in values),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_comparison(
    output_dir: str | Path,
    comparison: dict,
) -> tuple[Path, Path]:
    """Write deterministic ``comparison.json`` and ``comparison.md``."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "comparison.json"
    json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output / "comparison.md"
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.compare",
        description="Compare multiple ablation variant metrics.json reports",
    )
    parser.add_argument(
        "--variant-dirs",
        nargs="+",
        required=True,
        help="one directory per variant, each containing metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for comparison.json and comparison.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        variants = [load_variant_metrics(path) for path in args.variant_dirs]
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        print(f"comparison error: {exc}", file=sys.stderr)
        return 2
    comparison = build_comparison(variants)
    json_path, markdown_path = write_comparison(args.output_dir, comparison)
    print(f"comparison written to {json_path} and {markdown_path}")
    for row in comparison["rows"]:
        values = " | ".join(
            _format(row["values"][entry["variant"]])
            for entry in comparison["variants"]
        )
        print(f"  {row['label']}: {values}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
