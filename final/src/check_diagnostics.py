import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import WORK_ROOT, read_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="formal_20260930")
    parser.add_argument("--only-failed", action="store_true")
    args = parser.parse_args(argv)

    root = WORK_ROOT / "runs" / args.run_id
    if not root.is_dir():
        raise FileNotFoundError(f"run directory not found: {root}")
    rows = []
    for path in sorted(root.glob("*/*/diagnostics.json")):
        payload = read_json(path)
        relative = path.parent.relative_to(root)
        rows.append((str(relative), payload))
    if not rows:
        raise FileNotFoundError(f"no diagnostics under {root}")

    failures = 0
    for name, payload in rows:
        if payload.get("validation_only"):
            continue
        status = "PASS" if payload["passed"] else "FAIL"
        if not payload["passed"]:
            failures += 1
        if args.only_failed and payload["passed"]:
            continue
        print(
            f"{status} {name:14s} rhat {payload['max_rhat']:.5f} "
            f"ess_bulk {payload['min_ess_bulk']:7.1f} ess_tail {payload['min_ess_tail']:7.1f} "
            f"div {payload['divergences']} depth {payload.get('maxdepth_reached_sum', 0)} "
            f"mcse/sd {payload['max_mcse_sd_ratio']:.4f}"
        )
    print(f"\n{failures} failing run(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
