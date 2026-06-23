"""
Run LR-ASD for one synced interval video and write explicit success/error markers.

Usage:
    python lrasd_interval_worker.py <video_mp4> <repo_root> <model_path>
        [--success-name _SUCCESS]
        [--error-name asd_error.txt]
        [--stdout-name lrasd_stdout.log]
        [--stderr-name lrasd_stderr.log]

Given a synced video path like:
    .../synced_video/neural_23512014.mp4

LR-ASD writes outputs into:
    .../synced_video/neural_23512014/

This worker considers the run successful only if:
    1. Columbia_test.py exits cleanly
    2. output_dir/pyavi/video_out.avi exists and is non-empty
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path


def tail_text(path: Path, n_chars: int = 6000) -> str:
    if not path.exists():
        return "[missing]\n"
    text = path.read_text(errors="ignore")
    if len(text) <= n_chars:
        return text
    return "[truncated to last {} chars]\n".format(n_chars) + text[-n_chars:]


def derive_paths(video_mp4: Path) -> dict[str, Path]:
    video_folder = video_mp4.parent
    video_name = video_mp4.stem
    output_dir = video_folder / video_name
    ffmpeg_log = video_folder / f"{video_mp4.name}.ffmpeg.log"
    return {
        "video_folder": video_folder,
        "video_name": Path(video_name),
        "output_dir": output_dir,
        "wrapper_stdout": video_folder / f"{video_name}.lrasd_stdout.log",
        "wrapper_stderr": video_folder / f"{video_name}.lrasd_stderr.log",
        "wrapper_error": video_folder / f"{video_name}.asd_error.txt",
        "ffmpeg_log": ffmpeg_log,
        "pyavi_dir": output_dir / "pyavi",
        "video_out": output_dir / "pyavi" / "video_out.avi",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_mp4", type=Path)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--success-name", type=str, default="_SUCCESS")
    parser.add_argument("--error-name", type=str, default="asd_error.txt")
    parser.add_argument("--stdout-name", type=str, default="lrasd_stdout.log")
    parser.add_argument("--stderr-name", type=str, default="lrasd_stderr.log")
    args = parser.parse_args()

    video_mp4 = args.video_mp4
    repo_root = args.repo_root
    model_path = args.model_path

    if not video_mp4.exists():
        raise FileNotFoundError(f"Missing input video: {video_mp4}")
    if not repo_root.exists():
        raise FileNotFoundError(f"Missing LR-ASD repo root: {repo_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing LR-ASD model file: {model_path}")

    paths = derive_paths(video_mp4)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pyavi_dir = paths["pyavi_dir"]

    # LR-ASD itself creates `pyavi/` on success, so keep success markers there.
    # Error markers/logs should still be writable even when the model fails early
    # before creating that folder.
    success_path = pyavi_dir / args.success_name
    error_path = paths["wrapper_error"]
    stdout_path = paths["wrapper_stdout"]
    stderr_path = paths["wrapper_stderr"]

    if success_path.exists() and paths["video_out"].exists() and paths["video_out"].stat().st_size > 0:
        print(f"already done: {video_mp4}", flush=True)
        return 0

    if success_path.exists():
        success_path.unlink()
    if error_path.exists():
        error_path.unlink()

    if paths["ffmpeg_log"].exists():
        paths["ffmpeg_log"].unlink()

    cmd = [
        sys.executable,
        "-u",
        "Columbia_test.py",
        "--videoName",
        paths["video_name"].name,
        "--videoFolder",
        str(paths["video_folder"]),
        "--pretrainModel",
        str(model_path),
    ]

    try:
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        with open(stdout_path, "w") as stdout_f, open(stderr_path, "w") as stderr_f:
            result = subprocess.run(
                cmd,
                cwd=repo_root,
                env=child_env,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            print
            raise RuntimeError(f"Columbia_test.py failed with exit code {result.returncode}")

        if not paths["video_out"].exists():
            raise FileNotFoundError(f"LR-ASD finished but video_out.avi was not created: {paths['video_out']}")
        if paths["video_out"].stat().st_size == 0:
            raise RuntimeError(f"LR-ASD created empty video_out.avi: {paths['video_out']}")

        pyavi_dir.mkdir(parents=True, exist_ok=True)
        success_path.write_text("ok\n")
        print(f"done: {video_mp4}", flush=True)
        return 0

    except Exception:
        tb = traceback.format_exc()
        output_dir.mkdir(parents=True, exist_ok=True)
        parts = [
            "LR-ASD worker failure\n",
            "=" * 80 + "\n",
            f"video_mp4: {video_mp4}\n",
            f"repo_root: {repo_root}\n",
            f"model_path: {model_path}\n",
            f"output_dir: {output_dir}\n",
            f"stdout_log: {stdout_path}\n",
            f"stderr_log: {stderr_path}\n",
            f"ffmpeg_log: {paths['ffmpeg_log']}\n",
            f"expected_video_out: {paths['video_out']}\n",
            "\n",
            "Python traceback\n",
            "-" * 80 + "\n",
            tb,
            "\n",
            "LR-ASD stderr tail\n",
            "-" * 80 + "\n",
            tail_text(stderr_path),
            "\n",
            "LR-ASD stdout tail\n",
            "-" * 80 + "\n",
            tail_text(stdout_path),
            "\n",
            "ffmpeg log tail\n",
            "-" * 80 + "\n",
            tail_text(paths["ffmpeg_log"]),
        ]
        error_path.write_text("".join(parts))
        print(f"FAILED:\n{tb}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
