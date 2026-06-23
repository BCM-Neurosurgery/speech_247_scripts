"""
ASD / video-sync worker for a single convo-task patient.

Usage:
    python convo_asd_worker.py
        <patient_id>
        <stitched_root>    root of /mnt/stitched/EMU-18112 (patient subdir appended automatically)
        <video_dir>        e.g. /mnt/datalake/data/emu/YFCDatafile/VIDEO
        <out_dir>          e.g. anilu_comparison/YFC/video
        <repo_root>        video-sync-nbu-main checkout
        --task-keyword <keyword>   substring to select the convo task (e.g. 'EMU-0028_convo')
        [--cam-serial <serial> ...]
        [--log-level DEBUG]

Calls:
    python -m scripts.cli.cli_emu_time
        --patient-dir <stitched_root>/<patient_id>
        --video-dir   <video_dir>
        --out-dir     <out_dir>
        --keywords    <task_keyword>
        [--cam-serial ...]
        --log-level   <log_level>

Writes:
    <out_dir>/_SUCCESS  on success
    <out_dir>/_ERROR    on failure
    <out_dir>/asd_stdout.log
    <out_dir>/asd_stderr.log
"""

import argparse
import subprocess
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_id",    type=str,  help="Patient code, e.g. YFC")
    parser.add_argument("stitched_root", type=Path, help="Root of stitched mount, e.g. /mnt/stitched/EMU-18112")
    parser.add_argument("video_dir",     type=Path, help="Camera segment VIDEO directory")
    parser.add_argument("out_dir",       type=Path, help="Output directory for this patient")
    parser.add_argument("repo_root",     type=Path, help="video-sync-nbu-main checkout root")
    parser.add_argument("--task-keyword", required=True,
                        help="Substring to select the convo task (e.g. 'EMU-0028_convo')")
    parser.add_argument("--cam-serial", action="append", default=None,
                        help="Camera serial(s) to process; omit to process all cameras")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    args = parser.parse_args()

    patient_dir  = (args.stitched_root / args.patient_id).resolve()
    video_dir    = args.video_dir.resolve()
    out_dir      = args.out_dir.resolve()
    repo_root    = args.repo_root.resolve()
    success_path = out_dir / "_SUCCESS"
    error_path   = out_dir / "_ERROR"
    stdout_log   = out_dir / "asd_stdout.log"
    stderr_log   = out_dir / "asd_stderr.log"

    for label, p in [("stitched patient_dir", patient_dir),
                     ("video_dir",  video_dir),
                     ("repo_root",  repo_root)]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    success_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)

    python_bin = sys.executable
    cmd = [
        python_bin, "-u", "-m", "scripts.cli.cli_emu_time",
        "--patient-dir", str(patient_dir),
        "--video-dir",   str(video_dir),
        "--out-dir",     str(out_dir),
        "--keywords",    args.task_keyword,
        "--log-level",   args.log_level,
    ]
    if args.cam_serial:
        for serial in args.cam_serial:
            cmd += ["--cam-serial", serial]

    print(f"[START] {args.patient_id}  task={args.task_keyword}", flush=True)
    print(f"  cwd:  {repo_root}", flush=True)
    print(f"  cmd:  {' '.join(cmd)}", flush=True)

    try:
        with open(stdout_log, "w") as out_f, open(stderr_log, "w") as err_f:
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
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
        print(f"[OK] {args.patient_id}", flush=True)
        return 0

    except Exception:
        tb = traceback.format_exc()
        parts = [
            "convo_asd_worker failure\n",
            "=" * 80 + "\n",
            f"patient_id:   {args.patient_id}\n",
            f"patient_dir:  {patient_dir}\n",
            f"video_dir:    {video_dir}\n",
            f"out_dir:      {out_dir}\n",
            f"repo_root:    {repo_root}\n",
            f"task_keyword: {args.task_keyword}\n",
            f"stdout_log:   {stdout_log}\n",
            f"stderr_log:   {stderr_log}\n",
            "\nPython traceback\n",
            "-" * 80 + "\n",
            tb,
            "\ncli_emu_time stderr tail\n",
            "-" * 80 + "\n",
        ]
        if stderr_log.exists():
            text = stderr_log.read_text(errors="ignore")
            parts.append(text[-8000:] if len(text) > 8000 else text)
        parts += ["\ncli_emu_time stdout tail\n", "-" * 80 + "\n"]
        if stdout_log.exists():
            text = stdout_log.read_text(errors="ignore")
            parts.append(text[-4000:] if len(text) > 4000 else text)

        error_path.write_text("".join(parts))
        print(f"FAILED: {args.patient_id}\n{tb}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
