"""Command-line assembly for the shared import-check pipeline."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from meshengineer.curation.batch import open_report, run_import_check
from meshengineer.curation.manifest import read_manifest
from meshengineer.io.loader import read_storage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meshengineer")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("import-check", help="Check manifest assets and write a new JSONL report")
    for name in ("manifest", "storage", "report"):
        command.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        roots = read_storage(args.storage)
        refs = read_manifest(args.manifest, roots)
        stream = open_report(refs, roots, args.report, [args.manifest, args.storage])
    except (ValueError, OSError, UnicodeError) as error:
        print(f"Startup error: {error}", file=sys.stderr)
        return 2
    result = run_import_check(refs, roots, args.report, stream)
    print(json.dumps(result.to_json(), allow_nan=False))
    return 0 if result.completed and result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
