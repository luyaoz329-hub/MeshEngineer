"""Sequential import checks and non-overwriting, flushed JSONL reports."""

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import TextIO

from meshengineer.core.assets import AssetRef
from meshengineer.core.imports import BatchSummary
from meshengineer.pipeline.import_check import check_asset, environment_versions


def open_report(refs: Sequence[AssetRef], roots: Mapping[str, Path], report_path: Path,
                protected_paths: Sequence[Path] = ()) -> TextIO:
    report = report_path.resolve()
    if report_path.exists() or report_path.is_symlink():
        raise ValueError(f"Report must be a new file: {report_path}")
    for path in [*protected_paths, *(roots[r.source_id] / r.relative_path for r in refs)]:
        if report == path.resolve():
            raise ValueError(f"Report conflicts with an input: {path}")
    try:
        return report_path.open("x", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot create report: {error}") from error


def run_import_check(refs: Sequence[AssetRef], roots: Mapping[str, Path], report_path: Path,
                     stream: TextIO) -> BatchSummary:
    """Process a preflighted batch; unexpected exceptions keep their stack."""
    report = report_path.resolve()
    summary = BatchSummary(len(refs), str(report))
    environment = environment_versions()
    try:
        for ref in refs:
            record = check_asset(ref, roots, environment)
            summary.processed += 1
            if record.status == "ok": summary.ok += 1
            else: summary.failed += 1
            try:
                stream.write(json.dumps(record.to_json(), ensure_ascii=True, allow_nan=False) + "\n")
                stream.flush()
            except OSError as error:
                summary.abort_reason = f"Report write failed: {error}"
                break
            summary.reported += 1
            root = roots[ref.source_id]
            if record.code == "read_error" and (not root.is_dir() or not os.access(root, os.R_OK | os.X_OK)):
                summary.abort_reason = f"Storage root became unavailable: {root}"
                break
    finally:
        try:
            stream.close()
        except OSError as error:
            summary.abort_reason = f"Report close failed: {error}"
    summary.completed = summary.abort_reason is None and summary.processed == summary.reported == summary.total
    return summary
