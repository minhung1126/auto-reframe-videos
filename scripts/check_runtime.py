"""Fail GUI CI explicitly instead of allowing an entirely skipped GUI suite."""

import argparse
import json
from pathlib import Path
from auto_reframe_core.runtime_report import runtime_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gui", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = runtime_report(require_gui=args.require_gui)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
