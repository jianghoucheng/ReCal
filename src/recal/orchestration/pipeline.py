from __future__ import annotations

import argparse
import traceback

from recal.common import dump_json, utc_now
from recal.orchestration.manifests import initialize_experiment, write_summary
from recal.training.recovery import run_recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-stage", type=int, default=None)
    args = parser.parse_args()
    config, root = initialize_experiment(args.config, args.output_dir)
    status = {
        "started_at": utc_now(),
        "mode": "recovery",
        "status": "running",
        "last_successful_stage": None,
    }
    dump_json(status, root / "pipeline_status.json")
    try:
        final = run_recovery(
            config,
            root,
            dry_run=args.dry_run,
            start_stage=args.start_stage,
        )
        status["final_checkpoint"] = str(final)
        status.update(status="dry_run" if args.dry_run else "completed", finished_at=utc_now())
    except Exception as exc:
        status.update(
            status="failed",
            finished_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        dump_json(status, root / "pipeline_status.json")
        write_summary(root, {"pipeline_status": status["status"], "dry_run": args.dry_run})


if __name__ == "__main__":
    main()
