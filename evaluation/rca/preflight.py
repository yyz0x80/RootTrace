"""Evaluation preflight/dry-run: validate benchmark inputs without LLM calls.

Preflight verifies the manifest, public-dataset coverage, repo-cache
resolution, base-commit availability, gold-field isolation, historical
retrieval leakage, and the base-commit history boundary. It never invokes a
model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluation.rca.adapter import (
    FORBIDDEN_INPUT_FIELDS,
    build_incident_input,
    load_public_cases,
)
from evaluation.rca.leakage import load_history_instance_ids, validate_leakage
from evaluation.rca.manifest import load_manifest
from evaluation.rca.variants import AblationConfig, AblationVariant, variant_settings
from evaluation.rca.workspace import (
    create_case_workspace,
    destroy_case_workspace,
    resolve_repo_cache,
    verify_base_boundary,
)

FORBIDDEN_MANIFEST_FIELDS = ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")


def _scan_gold_fields(raw: dict) -> list[str]:
    """Find forbidden gold/PR fields anywhere in a raw manifest."""
    keys = set(raw.keys())
    for instance in raw.get("instances", []):
        if isinstance(instance, dict):
            keys.update(instance.keys())
    return sorted(key for key in keys if key in FORBIDDEN_MANIFEST_FIELDS)


def _git_cat_file_ok(mirror: Path, base_commit: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "cat-file", "-e", base_commit],
            cwd=mirror,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def run_preflight(args: argparse.Namespace) -> int:
    """Execute all preflight checks and print a JSON-summarizable report."""
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"preflight error: data root is not a directory: {data_root}", file=sys.stderr)
        return 2
    manifest_path = Path(
        args.manifest or data_root / "manifests" / "dev50.json"
    ).expanduser().resolve()
    smoke_path = Path(
        args.smoke_manifest or data_root / "manifests" / "smoke3.json"
    ).expanduser().resolve()
    repo_cache = Path(args.repo_cache or data_root / "repos").expanduser().resolve()
    public_path = Path(
        args.public or data_root / "public" / "verified_public.jsonl"
    ).expanduser().resolve()
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    # 1. Ablation config parses for the requested variant.
    try:
        settings = variant_settings(args.variant)
        config = AblationConfig(variant=args.variant)
        record(
            "ablation config",
            True,
            f"variant={settings.variant.value} hash={config.config_hash()[:12]}",
        )
    except (TypeError, ValueError) as exc:
        record("ablation config", False, str(exc))
        _finish(checks, args, manifest_path)
        return 2

    # 2. Manifest schema and gold-field isolation.
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record("manifest schema", False, f"cannot parse manifest: {exc}")
        _finish(checks, args, manifest_path)
        return 2
    try:
        manifest = load_manifest(manifest_path)
        record("manifest schema", True, f"{len(manifest.instances)} case(s)")
    except (TypeError, ValueError) as exc:
        record("manifest schema", False, str(exc))
        _finish(checks, args, manifest_path)
        return 2

    ids = [case.instance_id for case in manifest.instances]
    record(
        "unique instance ids",
        len(ids) == len(set(ids)),
        f"{len(ids)} unique",
    )
    gold_fields = _scan_gold_fields(raw)
    record(
        "manifest contains no gold fields",
        not gold_fields,
        f"found={gold_fields}" if gold_fields else "ok",
    )

    # 3. Public-dataset coverage and metadata consistency.
    try:
        public = load_public_cases(public_path)
    except (OSError, ValueError) as exc:
        record("public dataset", False, str(exc))
        _finish(checks, args, manifest_path)
        return 2
    missing = [case_id for case_id in ids if case_id not in public]
    record(
        "instances exist in public dataset",
        not missing,
        f"missing={sorted(missing)}" if missing else f"{len(public)} public case(s)",
    )
    mismatched = [
        case_id
        for case_id in ids
        if case_id in public
        and (public[case_id].repo, public[case_id].base_commit)
        != (
            next(case.repo for case in manifest.instances if case.instance_id == case_id),
            next(case.base_commit for case in manifest.instances if case.instance_id == case_id),
        )
    ]
    record(
        "manifest matches public metadata",
        not mismatched,
        f"mismatch={sorted(mismatched)}" if mismatched else "ok",
    )

    # 4. Repo-cache resolution and base-commit availability.
    cases = manifest.instances[: args.max_cases] if args.max_cases else manifest.instances
    mirrors: dict[str, Path] = {}
    resolve_ok = True
    for repo in sorted({case.repo for case in cases}):
        try:
            mirrors[repo] = resolve_repo_cache(repo_cache, repo)
        except FileNotFoundError as exc:
            record("repo cache resolves", False, str(exc))
            resolve_ok = False
    if resolve_ok:
        record(
            "repo cache resolves",
            True,
            ", ".join(f"{repo}={mirrors[repo].name}" for repo in sorted(mirrors)),
        )
    missing_commits = [
        case.instance_id
        for case in cases
        if case.repo in mirrors
        and not _git_cat_file_ok(mirrors[case.repo], case.base_commit)
    ]
    record(
        "base_commit exists in local mirror",
        not missing_commits,
        f"missing={sorted(missing_commits)}" if missing_commits else "all present",
    )

    # 5. Adapter isolation: RootTrace input never carries gold/test patch data.
    first_id = ids[0]
    adapter_forbidden: list[str] = []
    if first_id in public:
        incident = build_incident_input(public[first_id]).incident
        incident_keys = set(incident.model_dump().keys())
        adapter_forbidden = [
            key for key in FORBIDDEN_INPUT_FIELDS if key in incident_keys
        ]
    record(
        "adapter forwards no gold/test patch",
        not adapter_forbidden,
        f"found={adapter_forbidden}" if adapter_forbidden else "ok",
    )

    # 6. Historical-retrieval leakage guard.
    if args.history_corpus is not None:
        history_path = Path(args.history_corpus).expanduser().resolve()
        try:
            history_ids = load_history_instance_ids(history_path)
            targets = set(ids)
            if smoke_path.is_file():
                smoke = load_manifest(smoke_path)
                targets |= {case.instance_id for case in smoke.instances}
            leakage = validate_leakage(target_ids=targets, history_ids=history_ids)
            record(
                "evaluation targets excluded from historical retrieval",
                leakage["ok"],
                (
                    f"overlap={leakage['overlap_ids']}"
                    if not leakage["ok"]
                    else f"{len(history_ids)} history case(s), no overlap"
                ),
            )
        except (OSError, ValueError) as exc:
            record(
                "evaluation targets excluded from historical retrieval",
                False,
                str(exc),
            )
    else:
        record(
            "evaluation targets excluded from historical retrieval",
            True,
            "no history corpus configured; retrieval variants run without hints",
        )

    # 7. Git-history boundary: the workspace exposes only the base commit.
    if cases:
        first_case = cases[0]
        try:
            workspace = create_case_workspace(
                repo_cache,
                first_case,
                work_root=Path(tempfile.gettempdir()),
            )
            try:
                verify_base_boundary(workspace.repo, first_case.base_commit)
                count = subprocess.run(
                    ["git", "rev-list", "--all", "--count"],
                    cwd=workspace.repo,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                bounded = count.returncode == 0 and count.stdout.strip() == "1"
                record(
                    "git history bounded to base_commit",
                    bounded,
                    f"{first_case.instance_id} shallow workspace exposes 1 commit",
                )
            finally:
                destroy_case_workspace(workspace)
        except Exception as exc:  # noqa: BLE001 - preflight must surface any boundary failure
            record("git history bounded to base_commit", False, str(exc))
    else:
        record("git history bounded to base_commit", False, "manifest is empty")

    return _finish(checks, args, manifest_path)


def _finish(
    checks: list[dict],
    args: argparse.Namespace,
    manifest_path: Path,
) -> int:
    ok = all(check["ok"] for check in checks)
    report = {
        "schema_version": "1.0",
        "manifest": manifest_path.name,
        "variant": args.variant,
        "ok": ok,
        "checks": checks,
    }
    if args.json:
        output = Path(args.json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"preflight report written to {output}")
    for check in checks:
        marker = "PASS" if check["ok"] else "FAIL"
        print(f"[{marker}] {check['check']}: {check['detail']}")
    print(f"preflight {'passed' if ok else 'failed'}")
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    """Build the preflight CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.rca.preflight",
        description="RootTrace benchmark preflight (no LLM calls)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "Datasets" / "roottrace-swebench",
        help="SWE-bench data root (default: ~/Datasets/roottrace-swebench)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest to validate (default: <data-root>/manifests/dev50.json)",
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=None,
        help="smoke3 manifest used for leakage targets (default: <data-root>/manifests/smoke3.json)",
    )
    parser.add_argument(
        "--repo-cache",
        type=Path,
        default=None,
        help="local git mirror cache (default: <data-root>/repos)",
    )
    parser.add_argument(
        "--public",
        type=Path,
        default=None,
        help="public Verified dataset (default: <data-root>/public/verified_public.jsonl)",
    )
    parser.add_argument(
        "--history-corpus",
        type=Path,
        default=None,
        help="optional historical RCA corpus JSONL for the leakage guard",
    )
    parser.add_argument(
        "--variant",
        default="three_specialists_retrieval_off",
        choices=[variant.value for variant in AblationVariant],
        help="ablation variant to validate",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="limit base-commit checks to the first N manifest cases",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="optional path to write the preflight JSON report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_preflight(args)


if __name__ == "__main__":
    sys.exit(main())
