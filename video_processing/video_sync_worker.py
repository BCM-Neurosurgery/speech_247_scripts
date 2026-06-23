"""
Video-sync worker for a single interval.

Usage:
    python video_sync_worker.py <interval_dir> <repo_root> <video_dir> <cam_serial>
        [--keywords neural]
        [--log-level DEBUG]

Runs:
    python -m scripts.cli.cli_emu_time
        --patient-dir <interval_dir>
        --video-dir   <video_dir>
        --out-dir     <interval_dir>/video
        --keywords    <keywords>
        --cam-serial  <cam_serial>
        --log-level   <log_level>

Writes:
    <interval_dir>/video/<interval_id>/_SUCCESS  (on success)
    <interval_dir>/video/<interval_id>/_ERROR    (on failure)
"""

import argparse
import subprocess
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("interval_dir", type=Path)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("cam_serial", type=str)
    parser.add_argument("--keywords", type=str, default="neural")
    parser.add_argument("--log-level", type=str, default="DEBUG")
    args = parser.parse_args()

    interval_dir = args.interval_dir.resolve()
    interval_id = interval_dir.name
    out_dir = interval_dir / "video"
    pipeline_out_dir = out_dir / interval_id
    success_path = pipeline_out_dir / "_SUCCESS"
    error_path = pipeline_out_dir / "_ERROR"
    stdout_log = pipeline_out_dir / "video_sync_stdout.log"
    stderr_log = pipeline_out_dir / "video_sync_stderr.log"

    if not interval_dir.exists():
        print(f"ERROR: interval_dir does not exist: {interval_dir}", file=sys.stderr)
        return 1
    if not args.repo_root.exists():
        print(f"ERROR: repo_root does not exist: {args.repo_root}", file=sys.stderr)
        return 1
    if not args.video_dir.exists():
        print(f"ERROR: video_dir does not exist: {args.video_dir}", file=sys.stderr)
        return 1

    pipeline_out_dir.mkdir(parents=True, exist_ok=True)
    success_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "scripts.cli.cli_emu_time",
        "--patient-dir", str(interval_dir),
        "--video-dir",   str(args.video_dir),
        "--out-dir",     str(out_dir),
        "--keywords",    args.keywords,
        "--cam-serial",  args.cam_serial,
        "--log-level",   args.log_level,
    ]

    print(f"[START] {interval_id}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    try:
        with open(stdout_log, "w") as out_f, open(stderr_log, "w") as err_f:
            result = subprocess.run(
                cmd,
                cwd=str(args.repo_root),
                stdout=out_f,
                stderr=err_f,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"cli_emu_time exited with code {result.returncode}. "
                f"See {stderr_log}"
            )

        success_path.write_text("ok\n")
        print(f"[OK] {interval_id}", flush=True)
        return 0

    except Exception:
        tb = traceback.format_exc()
        parts = [
            f"video_sync_worker failure\n",
            "=" * 80 + "\n",
            f"interval_dir: {interval_dir}\n",
            f"repo_root:    {args.repo_root}\n",
            f"video_dir:    {args.video_dir}\n",
            f"cam_serial:   {args.cam_serial}\n",
            f"stdout_log:   {stdout_log}\n",
            f"stderr_log:   {stderr_log}\n",
            "\n",
            "Python traceback\n",
            "-" * 80 + "\n",
            tb,
            "\n",
            "cli_emu_time stderr tail\n",
            "-" * 80 + "\n",
        ]
        if stderr_log.exists():
            text = stderr_log.read_text(errors="ignore")
            parts.append(text[-6000:] if len(text) > 6000 else text)
        parts += [
            "\n",
            "cli_emu_time stdout tail\n",
            "-" * 80 + "\n",
        ]
        if stdout_log.exists():
            text = stdout_log.read_text(errors="ignore")
            parts.append(text[-6000:] if len(text) > 6000 else text)

        error_path.write_text("".join(parts))
        print(f"FAILED: {interval_id}\n{tb}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
