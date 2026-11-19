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
import subprocess
import sys
import time
from typing import List, Optional

from aix_zero.meta_ai_autopilot import (
    AutonomousSelfImprovementController,
    DEFAULT_STATE_DIR,
    analyze_python_code_quality_and_ast,
    export_peer_ai_coaching_pack,
    grade_and_merge_python_patches,
    harvest_external_open_source_projects,
    normalize_and_verify_streamed_rows,
    solve_python_repo_task_and_emit_diff,
    stream_huggingface_open_source_rows,
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
    p_once.add_argument("--auto-push", action="store_true", help="Automatically commit and push verified shards/telemetry")

    p_start = sub.add_parser("start", help="Run unattended self-improvement loop until stopped")
    p_start.add_argument("--max-cycles", type=int, default=1000, help="Maximum cycles before pausing")
    p_start.add_argument("--interval-seconds", "--interval", dest="interval_seconds", type=float, default=2.0, help="Sleep interval between cycles")
    p_start.add_argument("--threads", type=int, default=1, help="Max CPU threads (default: 1)")
    p_start.add_argument("--sync-github", action="store_true", help="Open and close auditable GitHub issues")
    p_start.add_argument("--auto-push", action="store_true", help="Automatically commit and push verified shards/telemetry")

    p_daemon = sub.add_parser("start-daemon", help="Launch detached background self-improvement daemon and return immediately")
    p_daemon.add_argument("--max-cycles", type=int, default=1000, help="Maximum cycles before pausing")
    p_daemon.add_argument("--interval-seconds", "--interval", dest="interval_seconds", type=float, default=120.0, help="Sleep interval between cycles")
    p_daemon.add_argument("--threads", type=int, default=1, help="Max CPU threads (default: 1)")
    p_daemon.add_argument("--sync-github", action="store_true", help="Open and close auditable GitHub issues")
    p_daemon.add_argument("--auto-push", action="store_true", help="Automatically commit and push verified shards/telemetry")

    p_query = sub.add_parser("query", help="Query the evolving v12 Meta-AI model on a math or Meta-AI prompt")
    p_query.add_argument("--prompt", required=True, help="Prompt string (e.g. 'Q:MetaScale:C=144;N=4->D_opt?' or '14*23=')")
    p_query.add_argument("--no-alu", action="store_true", help="Disable neuro-symbolic ALU assistance")
    p_query.add_argument("--threads", type=int, default=1, help="Max CPU threads (default: 1)")

    p_coach = sub.add_parser("coach", help="Grade a conversing peer AI's response and generate corrective coaching feedback")
    p_coach.add_argument("--prompt", required=True, help="Meta-AI prompt string (e.g. 'Q:MetaRLVR:r=1;successes=2;group=8->grpo_advantage?')")
    p_coach.add_argument("--peer-response", required=True, help="Peer AI response containing <think>...</think><ans>...</ans>")

    p_pack = sub.add_parser("export-coaching-pack", help="Export portable 12-domain Peer-AI Symbiotic Coaching Pack JSON")
    p_pack.add_argument("--cycle", type=int, default=1, help="Cycle index for worked examples")

    p_backlog = sub.add_parser("backlog", help="Discover and inspect autonomous self-improvement backlog issues")
    p_backlog.add_argument("--sync-github", action="store_true", help="Open GitHub issues for newly discovered backlog items")

    p_harvest = sub.add_parser("harvest-external", help="Catalog and optionally shallow-clone external open-source AI repositories")
    p_harvest.add_argument("--clone", nargs="*", default=[], help="Optional GitHub repo slugs to shallow-clone into open_source_corpus/")

    p_peer = sub.add_parser("peer-session", help="Run a multi-turn Peer-AI Symbiosis coaching & mutual-distillation session")
    p_peer.add_argument("--endpoint", default=None, help="Optional Ollama or OpenAI-compatible HTTP endpoint URL")
    p_peer.add_argument("--model", default="simulated-peer-ai", help="Peer AI model name")
    p_peer.add_argument("--max-prompts", type=int, default=4, help="Number of prompts to evaluate and coach")

    p_stream = sub.add_parser("stream-hf", help="Stream and R1-verify open-source training rows from Hugging Face Datasets Server")
    p_stream.add_argument("--dataset", default="open-r1/OpenR1-Math-220k", help="Hugging Face dataset slug")
    p_stream.add_argument("--offset", type=int, default=0, help="Row offset")
    p_stream.add_argument("--limit", type=int, default=4, help="Number of rows to stream")
    p_stream.add_argument("--offline", action="store_true", help="Force deterministic offline fallback")

    p_arena = sub.add_parser("arena", help="Run 20-Domain Frontier Benchmark Arena & SWE patch verification scorecard")
    p_arena.add_argument("--cycle", type=int, default=1, help="Cycle index for benchmark generation")

    p_py = sub.add_parser("py-analyze", help="Analyze Python source file or inline snippet with AST & bytecode disassembler")
    p_py.add_argument("--file", default=None, help="Path to Python file to analyze")
    p_py.add_argument("--code", default=None, help="Inline Python code string to analyze")

    sub.add_parser("solve-task", help="Verify SparseExpertCartridgeBank.execute_routed_request contract and emit unified diff")
    p_grade = sub.add_parser("grade-patch", help="Grade AIX, peer, and collaborative patches on the routed-request contract suite")
    p_grade.add_argument("--aix", default=None, help="Optional path to AIX patch file")
    p_grade.add_argument("--peer", default=None, help="Optional path to peer patch file")
    p_grade.add_argument("--collaborative", default=None, help="Optional path to collaborative patch file")
    p_grade.add_argument("--reference", default=None, help="Optional path to reference patch file")

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
        res = ctrl.run_single_self_improvement_cycle(
            sync_github_issue=args.sync_github,
            auto_push=args.auto_push,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "start":
        ctrl.clear_stop_sentinel()
        ctrl.install_signal_handlers()
        completed = 0
        last_res = {}
        while completed < args.max_cycles and not ctrl.should_stop():
            last_res = ctrl.run_single_self_improvement_cycle(
                sync_github_issue=args.sync_github,
                auto_push=args.auto_push,
            )
            if not last_res.get("executed", False):
                break
            completed += 1
            if completed < args.max_cycles and not ctrl.should_stop():
                time.sleep(max(0.0, float(args.interval_seconds)))
        print(json.dumps(ctrl.get_status(), indent=2))
        return 0

    if args.command == "start-daemon":
        ctrl.clear_stop_sentinel()
        ctrl.state_dir.mkdir(parents=True, exist_ok=True)
        log_path = ctrl.state_dir / "autopilot_daemon.log"
        cmd = [
            sys.executable,
            "-m",
            "aix_zero.aix_ctl",
            "--state-dir",
            str(ctrl.state_dir),
            "start",
            "--max-cycles",
            str(args.max_cycles),
            "--interval-seconds",
            str(args.interval_seconds),
            "--threads",
            str(args.threads),
        ]
        if args.sync_github:
            cmd.append("--sync-github")
        if args.auto_push:
            cmd.append("--auto-push")
        with open(log_path, "a", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ctrl.repo_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        ctrl.pid_path.write_text(str(proc.pid), encoding="utf-8")
        print(
            json.dumps(
                {
                    "daemon_started": True,
                    "pid": proc.pid,
                    "max_cpu_threads": args.threads,
                    "interval_seconds": args.interval_seconds,
                    "sync_github": bool(args.sync_github),
                    "auto_push": bool(args.auto_push),
                    "log_path": str(log_path),
                    "stop_command": f"{sys.executable} -m aix_zero.aix_ctl stop",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "query":
        res = ctrl.query_evolving_meta_ai(args.prompt, use_alu=not args.no_alu)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "coach":
        res = ctrl.coach_peer_ai_response(args.prompt, args.peer_response)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "export-coaching-pack":
        pack = export_peer_ai_coaching_pack(ctrl.state_dir / "peer_ai_coaching_pack.json", cycle_index=args.cycle)
        print(json.dumps(pack, indent=2))
        return 0

    if args.command == "backlog":
        res = ctrl.discover_and_file_improvement_issues(sync_github=args.sync_github)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "harvest-external":
        res = harvest_external_open_source_projects(
            ctrl.repo_root,
            clone_slugs=args.clone,
            output_path=ctrl.state_dir / "external_open_source_harvest.json",
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "peer-session":
        res = ctrl.run_peer_ai_symbiosis_session(
            endpoint_url=args.endpoint,
            model_name=args.model,
            max_prompts=args.max_prompts,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "stream-hf":
        raw = stream_huggingface_open_source_rows(
            args.dataset,
            offset=args.offset,
            limit=args.limit,
            allow_network=not args.offline,
        )
        verified = normalize_and_verify_streamed_rows(
            raw,
            output_path=ctrl.state_dir / "streamed_hf_verified_batch.json",
        )
        print(json.dumps(verified, indent=2))
        return 0

    if args.command == "arena":
        res = ctrl.run_frontier_arena_evaluation(
            cycle_index=args.cycle,
            output_path=ctrl.state_dir / "frontier_arena_scorecard.json",
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "py-analyze":
        if args.file:
            res = analyze_python_code_quality_and_ast(args.file, is_file_path=True)
        else:
            res = analyze_python_code_quality_and_ast(args.code or "def f(x: int) -> int:\n    return x + 1\n", is_file_path=False)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "solve-task":
        res = solve_python_repo_task_and_emit_diff(ctrl.repo_root)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "grade-patch":
        res = grade_and_merge_python_patches(ctrl.repo_root)
        print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
