from __future__ import annotations

import json

from evaluation.suite import run_evaluation


def main() -> None:
    report = run_evaluation()
    print(json.dumps({"passed": report["passed_experiments"], "total": report["total_experiments"]}, indent=2))


if __name__ == "__main__":
    main()
