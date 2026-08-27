"""Reproducible dev50 manifest selection.

Selection is fixed and fully deterministic:

1. read every candidate from ``public/verified_public.jsonl``;
2. exclude the smoke3 instance ids;
3. validate required fields and instance-id uniqueness;
4. stratify proportionally by repo;
5. allocate per-repo quotas with the Hamilton largest-remainder method so the
   total is strictly the requested size;
6. inside each repo, sort candidates by ``SHA256(f"{seed}:{instance_id}")``
   and take the repo's top quota;
7. emit the final manifest sorted by ``instance_id``;
8. fixed seed = 42;
9. ``difficulty`` never influences selection; it is recorded only as a
   diagnostic distribution for source/dev50.

Manifest case entries contain only ``instance_id``, ``repo``, and
``base_commit``. Gold/test patch and FAIL_TO_PASS/PASS_TO_PASS metadata are
never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from evaluation.adapter import PublicCase, load_public_cases
from evaluation.leakage import load_history_instance_ids, validate_leakage
from evaluation.manifest import ManifestCase, RcaManifest, load_manifest

SELECTOR_NAME = "evaluation.select_manifest"
SELECTOR_VERSION = "1.0"
SELECTION_METHOD = "proportional-stratified-hamilton-largest-remainder"
SORT_KEY_METHOD = 'sha256(f"{seed}:{instance_id}")'
DEFAULT_SEED = 42
DEFAULT_SIZE = 50
DEFAULT_MANIFEST_NAME = "dev50"


def _selection_rank(seed: int, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def allocate_quotas(repo_counts: dict[str, int], size: int) -> dict[str, int]:
    """Hamilton largest-remainder allocation; total always equals ``size``."""
    if size <= 0:
        raise ValueError("requested size must be positive")
    total = sum(repo_counts.values())
    if total <= 0:
        raise ValueError("no eligible cases to allocate")
    if size > total:
        raise ValueError(
            f"requested size {size} exceeds eligible count {total}"
        )

    quotas: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for repo in sorted(repo_counts):
        exact = repo_counts[repo] * size / total
        quotas[repo] = int(exact)
        remainders[repo] = exact - quotas[repo]

    remaining = size - sum(quotas.values())
    # Largest remainder first; ties break by repo name for byte-stability.
    ordered = sorted(remainders, key=lambda repo: (-remainders[repo], repo))
    for repo in ordered[:remaining]:
        quotas[repo] += 1
    return quotas


def select_cases(
    public_cases: list[PublicCase],
    excluded_ids: set[str],
    *,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_SIZE,
) -> list[PublicCase]:
    """Select ``size`` cases deterministically (proportional by repo)."""
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seen: set[str] = set()
    eligible: list[PublicCase] = []
    for case in public_cases:
        if case.instance_id in seen:
            raise ValueError(f"duplicate instance_id: {case.instance_id}")
        seen.add(case.instance_id)
        if case.instance_id not in excluded_ids:
            eligible.append(case)
    if len(eligible) < size:
        raise ValueError(
            f"only {len(eligible)} eligible cases; requested {size}"
        )

    repo_counts = Counter(case.repo for case in eligible)
    quotas = allocate_quotas(dict(repo_counts), size)
    selected: list[PublicCase] = []
    for repo in sorted(quotas):
        quota = quotas[repo]
        if quota <= 0:
            continue
        pool = sorted(
            (case for case in eligible if case.repo == repo),
            key=lambda case: _selection_rank(seed, case.instance_id),
        )
        selected.extend(pool[:quota])

    selected.sort(key=lambda case: case.instance_id)
    selected_ids = [case.instance_id for case in selected]
    if len(selected_ids) != size:
        raise ValueError(f"selection size mismatch: {len(selected_ids)} != {size}")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selection contains duplicate instance ids")
    if set(selected_ids) & excluded_ids:
        raise ValueError("selection overlaps excluded smoke3 ids")
    if not set(selected_ids) <= seen:
        raise ValueError("selection references unknown instance ids")
    return selected


def build_dev50_manifest(
    selected: list[PublicCase],
    *,
    name: str = DEFAULT_MANIFEST_NAME,
    seed: int = DEFAULT_SEED,
) -> RcaManifest:
    """Build the manifest containing only instance_id/repo/base_commit."""
    instances = [
        ManifestCase(
            instance_id=case.instance_id,
            repo=case.repo,
            base_commit=case.base_commit,
        )
        for case in selected
    ]
    return RcaManifest(name=name, seed=seed, instances=instances)


def manifest_bytes(manifest: RcaManifest) -> bytes:
    """Deterministic canonical serialization for byte-stable output."""
    return (
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _difficulty_distribution(cases: list[PublicCase]) -> dict[str, int]:
    distribution: Counter[str] = Counter()
    for case in cases:
        distribution[case.difficulty or "(unknown)"] += 1
    return dict(sorted(distribution.items()))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_selection_report(
    *,
    selected: list[PublicCase],
    eligible: list[PublicCase],
    excluded_ids: set[str],
    quotas: dict[str, int],
    public_count: int,
    source_path: Path,
    smoke_manifest_path: Path,
    dev50_bytes: bytes,
    seed: int,
    size: int,
    leakage: dict | None,
) -> dict:
    """Build the deterministic selection report."""
    selected_by_repo: dict[str, list[str]] = {}
    for case in selected:
        selected_by_repo.setdefault(case.repo, []).append(case.instance_id)
    source_by_repo = Counter(case.repo for case in eligible)
    per_repo = []
    for repo in sorted(source_by_repo):
        per_repo.append(
            {
                "repo": repo,
                "source_count": source_by_repo[repo],
                "allocated_quota": quotas.get(repo, 0),
                "selected_count": len(selected_by_repo.get(repo, [])),
                "selected": selected_by_repo.get(repo, []),
            }
        )
    report = {
        "schema_version": "1.0",
        "selector": {
            "name": SELECTOR_NAME,
            "version": SELECTOR_VERSION,
            "method": SELECTION_METHOD,
            "sort_key": SORT_KEY_METHOD,
        },
        "config": {
            "seed": seed,
            "requested_size": size,
            "source_dataset": source_path.name,
            "smoke_manifest": smoke_manifest_path.name,
        },
        "counts": {
            "source_dataset_count": public_count,
            "excluded_smoke_count": len(excluded_ids),
            "eligible_count": len(eligible),
            "selected_count": len(selected),
        },
        "per_repo": per_repo,
        "difficulty_distribution": {
            "source": _difficulty_distribution(eligible),
            "dev50": _difficulty_distribution(selected),
        },
        "hashes": {
            "source_dataset_sha256": _file_sha256(source_path),
            "smoke3_manifest_sha256": _file_sha256(smoke_manifest_path),
            "dev50_manifest_sha256": hashlib.sha256(dev50_bytes).hexdigest(),
        },
        "leakage_validation": leakage,
    }
    return report


def run_selection(args: argparse.Namespace) -> int:
    """Run the dev50 manifest selection from parsed CLI arguments."""
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"selection error: data root is not a directory: {data_root}", file=sys.stderr)
        return 2
    source_path = Path(args.source or data_root / "public" / "verified_public.jsonl").expanduser().resolve()
    smoke_manifest_path = Path(
        args.smoke_manifest or data_root / "manifests" / "smoke3.json"
    ).expanduser().resolve()
    output_dir = Path(args.output_dir or data_root / "manifests").expanduser().resolve()
    if args.size <= 0:
        print("selection error: --size must be positive", file=sys.stderr)
        return 2
    if args.seed < 0:
        print("selection error: --seed must be non-negative", file=sys.stderr)
        return 2
    for label, path in (
        ("source dataset", source_path),
        ("smoke3 manifest", smoke_manifest_path),
    ):
        if not path.is_file():
            print(f"selection error: {label} file not found: {path}", file=sys.stderr)
            return 2
    if args.history_corpus is not None:
        history_path = Path(args.history_corpus).expanduser().resolve()
        if not history_path.is_file():
            print(f"selection error: history corpus not found: {history_path}", file=sys.stderr)
            return 2

    try:
        public = load_public_cases(source_path)
        smoke = load_manifest(smoke_manifest_path)
    except (TypeError, ValueError) as exc:
        print(f"selection error: {exc}", file=sys.stderr)
        return 2

    excluded_ids = {case.instance_id for case in smoke.instances}
    public_cases = list(public.values())
    eligible = [case for case in public_cases if case.instance_id not in excluded_ids]
    try:
        selected = select_cases(
            public_cases,
            excluded_ids,
            seed=args.seed,
            size=args.size,
        )
    except ValueError as exc:
        print(f"selection error: {exc}", file=sys.stderr)
        return 2

    manifest = build_dev50_manifest(selected, name=args.manifest_name, seed=args.seed)
    payload = manifest_bytes(manifest)
    quotas = allocate_quotas(
        dict(Counter(case.repo for case in eligible)),
        args.size,
    )

    leakage: dict | None = None
    if args.history_corpus is not None:
        try:
            history_ids = load_history_instance_ids(args.history_corpus)
            target_ids = excluded_ids | {case.instance_id for case in selected}
            leakage = validate_leakage(target_ids=target_ids, history_ids=history_ids)
        except ValueError as exc:
            print(f"selection error: {exc}", file=sys.stderr)
            return 2
        if leakage and not leakage["ok"]:
            print(
                f"selection error: leakage with historical corpus: "
                f"{leakage['overlap_ids']}",
                file=sys.stderr,
            )
            return 2

    report = build_selection_report(
        selected=selected,
        eligible=eligible,
        excluded_ids=excluded_ids,
        quotas=quotas,
        public_count=len(public_cases),
        source_path=source_path,
        smoke_manifest_path=smoke_manifest_path,
        dev50_bytes=payload,
        seed=args.seed,
        size=args.size,
        leakage=leakage,
    )

    if args.dry_run:
        print(f"DRY RUN: would write {args.size} case(s) to {output_dir}")
        return 0

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{args.manifest_name}.json"
        report_path = output_dir / f"{args.manifest_name}_selection_report.json"
        manifest_path.write_bytes(payload)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"selection error: cannot write output: {exc}", file=sys.stderr)
        return 2

    print(
        f"selection complete: {len(selected)} case(s) "
        f"-> {manifest_path} (+ selection report)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the dev50 manifest selector CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.select_manifest",
        description="Reproducible dev50 manifest selection",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "Datasets" / "roottrace-swebench",
        help="SWE-bench data root (default: ~/Datasets/roottrace-swebench)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="public Verified dataset (default: <data-root>/public/verified_public.jsonl)",
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=None,
        help="smoke3 manifest to exclude (default: <data-root>/manifests/smoke3.json)",
    )
    parser.add_argument(
        "--history-corpus",
        type=Path,
        default=None,
        help="optional historical RCA corpus JSONL; overlap with targets is rejected",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <data-root>/manifests)",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help="manifest base name (default: dev50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="fixed selection seed (default: 42)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="requested manifest size (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and selection without writing files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_selection(args)


if __name__ == "__main__":
    sys.exit(main())
