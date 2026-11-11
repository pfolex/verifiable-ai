#!/usr/bin/env python3
"""
AIX-Zero Autopilot Control CLI (`aix_zero/aix_ctl.py`)
======================================================
Provides one-command start, status, and immediate shutdown/kill-switch controls for the
Meta-AI Autonomous Self-Improvement Engine:

- Turn OFF immediately (safe for shutting down your computer):
    python -m aix_zero.aix_ctl stop

- Check live status:
    python -m aix_zero.aix_ctl status

- Run one self-contained Meta-AI improvement cycle (optionally opening/closing a GitHub issue):
    python -m aix_zero.aix_ctl run-once [--sync-github]

- Run unattended loop (stops automatically when `aix_ctl stop` is called):
    python -m aix_zero.aix_ctl start --max-cycles 5 --interval-seconds 10 --threads 1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import List, Optional

from aix_zero.meta_ai_autopilot import (
    AutonomousSelfImprovementController,
    DEFAULT_STATE_DIR,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AIX-Zero Meta-AI Autopilot Control CLI")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Directory storing autopilot status, PID, and stop sentinel",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stop = sub.add_parser("stop", help="Immediately stop the autopilot so you can shut down your machine")
    p_stop.add_argument("--reason", default="USER_SHUTDOWN_REQUEST", help="Reason recorded in status log")

    sub.add_parser("status", help="Show current autopilot state and completed Meta-AI cycles")

    p_once = sub.add_parser("run-once", help="Run a single Meta-AI self-improvement cycle")
    p_once.add_argument("--threads", type=int, default=1, help="Max CPU threads (default: 1)")
    p_once.add_argument("--sync-github", action="store_true", help="Open and close an auditable GitHub issue")

    p_start = sub.add_parser("start", help="Run unattended self-improvement loop until stopped")
    p_start.add_argument("--max-cycles", type=int, default=3, help="Maximum cycles before pausing")
    p_start.add_argument("--interval-seconds", type=float, default=2.0, help="Sleep interval between cycles")
    p_start.add_argument("--threads", type=int, default=1, help="Max CPU threads (default: 1)")
    p_start.add_argument("--sync-github", action="store_true", help="Open and close auditable GitHub issues")

    args = parser.parse_args(argv)
    ctrl = AutonomousSelfImprovementController(
        args.state_dir,
        max_cpu_threads=getattr(args, "threads", 1),
    )

    if args.command == "stop":
        # Send SIGTERM to active PID if present and alive
        if ctrl.pid_path.exists():
            try:
                pid = int(ctrl.pid_path.read_text(encoding="utf-8").strip())
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        st = ctrl.request_stop(reason=args.reason)
        print(json.dumps(st, indent=2))
        return 0

    if args.command == "status":
        print(json.dumps(ctrl.get_status(), indent=2))
        return 0

    if args.command == "run-once":
        ctrl.clear_stop_sentinel()
        ctrl.install_signal_handlers()
        res = ctrl.run_single_self_improvement_cycle(sync_github_issue=args.sync_github)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "start":
        ctrl.clear_stop_sentinel()
        ctrl.install_signal_handlers()
        completed = 0
        last_res = {}
        while completed < args.max_cycles and not ctrl.should_stop():
            last_res = ctrl.run_single_self_improvement_cycle(sync_github_issue=args.sync_github)
            if not last_res.get("executed", False):
                break
            completed += 1
            if completed < args.max_cycles and not ctrl.should_stop():
                time.sleep(max(0.0, float(args.interval_seconds)))
        print(json.dumps(ctrl.get_status(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
