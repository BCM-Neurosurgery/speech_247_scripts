"""
Reformat copied legacy EMU video folders for video-sync-nbu.

This script is intended for copied VIDEO trees such as:

    /mnt/projectworlds/EMU-18112/YFA_Datafile/VIDEO

It does not need write access to datalake. By default it is a dry run. Pass
--execute to apply changes.

The video-sync-nbu EMU time pipeline expects:

    VIDEO/
      YYYYMMDD/
        <SEG>.json
        <SEG>.<CAM>.mp4

Legacy continuous files often look like:

    <BASE>.<CAM>_<BLOCK>.mp4

with a single <BASE>.json covering all numbered blocks. This script splits the
legacy JSON into per-block JSON files and renames/moves the MP4s into the flat
date directory expected by video-sync-nbu.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


LEGACY_MP4_RE = re.compile(
    r"^(?P<base>.+)\.(?P<cam>[0-9A-Za-z]+)_(?P<block>\d+)\.mp4$",
    re.IGNORECASE,
)
NEW_MP4_RE = re.compile(
    r"^(?P<seg>.+)\.(?P<cam>[0-9A-Za-z]+)\.mp4$",
    re.IGNORECASE,
)
DATE_DIR_RE = re.compile(r"^\d{8}$")
BASE_TIME_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")

LIST_FIELDS = {
    "real_times",
    "timestamps",
    "frame_id",
    "frame_id_abs",
    "chunk_serial_data",
    "serial_msg",
}


@dataclass(frozen=True)
class Operation:
    action: str
    src: str | None
    dst: str | None
    reason: str


@dataclass
class LegacyVideo:
    path: Path
    base: str
    cam: str
    block: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reformat copied legacy EMU VIDEO data for video-sync-nbu."
    )
    parser.add_argument(
        "--video-root",
        required=True,
        type=Path,
        help="Copied VIDEO directory to reformat, e.g. /mnt/projectworlds/EMU-18112/YFA_Datafile/VIDEO",
    )
    parser.add_argument(
        "--patient",
        choices=["YEY", "YEZ", "YFA"],
        default=None,
        help="Optional safety check for the patient being reformatted.",
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=18000,
        help="Frames per legacy numbered block. Default: 18000 (10 min at 30 fps).",
    )
    parser.add_argument(
        "--filename-timezone",
        default="America/Chicago",
        help=(
            "Timezone used for generated filename timestamps. JSON real_times are UTC; "
            "new acquisition filenames are local Central time by default."
        ),
    )
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=30.0,
        help="Skip files modified more recently than this, to avoid active copies. Use 0 after copying is complete.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes. Without this, only prints/writes a dry-run manifest.",
    )
    parser.add_argument(
        "--quarantine-orphans",
        action="store_true",
        help=(
            "Move legacy MP4s with no matching JSON beside VIDEO into VIDEO_legacy_unusable. "
            "This can make VIDEO strict-compatible, but those videos cannot be time-synced "
            "unless a companion JSON is found. Default: leave them in place."
        ),
    )
    parser.add_argument(
        "--delete-quarantine",
        action="store_true",
        help="Delete VIDEO_legacy_unusable after quarantining. Requires --execute --quarantine-orphans.",
    )
    parser.add_argument(
        "--keep-original-json",
        action="store_true",
        help=(
            "Leave original multi-block JSON files in place. By default they move beside "
            "VIDEO into VIDEO_legacy_originals so VIDEO remains pipeline-compatible."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for the operation manifest JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def is_recent(path: Path, min_age_minutes: float) -> bool:
    if min_age_minutes <= 0:
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < min_age_minutes * 60.0


def date_dir_for(path: Path, fallback_seg: str) -> str:
    for parent in [path.parent, *path.parents]:
        if DATE_DIR_RE.fullmatch(parent.name):
            return parent.name
    match = BASE_TIME_RE.search(fallback_seg)
    if match:
        return match.group("date")
    raise ValueError(f"Could not determine YYYYMMDD date folder for {path}")


def infer_segment_id(base: str, block: int, split_payload: dict[str, Any], filename_tz: ZoneInfo) -> str:
    real_times = split_payload.get("real_times") or []
    if real_times:
        start_utc = datetime.strptime(real_times[0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        start = start_utc.astimezone(filename_tz)
        stem_root = root_without_tail_timestamp(base)
        return f"{stem_root}_{start.strftime('%Y%m%d_%H%M%S')}"

    # Fallback for metadata without real_times. This should be rare and is less
    # accurate, but keeps the output sortable and deterministic.
    match = BASE_TIME_RE.search(base)
    if not match:
        return f"{base}_block{block:04d}"
    return base if block == 0 else f"{base}_block{block:04d}"


def root_without_tail_timestamp(base: str) -> str:
    parts = base.split("_")
    if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[-2]) and re.fullmatch(r"\d{6}", parts[-1]):
        return "_".join(parts[:-2])
    return base


def slice_json(payload: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in LIST_FIELDS and isinstance(value, list):
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def sidecar_dir(video_root: Path, suffix: str) -> Path:
    return video_root.parent / f"{video_root.name}_{suffix}"


def find_legacy_videos(video_root: Path, min_age_minutes: float) -> tuple[list[LegacyVideo], list[Path]]:
    legacy: list[LegacyVideo] = []
    skipped_recent: list[Path] = []
    for mp4 in video_root.rglob("*.mp4"):
        if is_recent(mp4, min_age_minutes):
            skipped_recent.append(mp4)
            continue
        match = LEGACY_MP4_RE.match(mp4.name)
        if not match:
            continue
        legacy.append(
            LegacyVideo(
                path=mp4,
                base=match.group("base"),
                cam=match.group("cam"),
                block=int(match.group("block")),
            )
        )
    return legacy, skipped_recent


def find_json_for_base(video_root: Path, base: str, preferred_dir: Path) -> Path | None:
    direct = preferred_dir / f"{base}.json"
    if direct.exists():
        return direct

    matches = list(video_root.rglob(f"{base}.json"))
    if not matches:
        return None
    matches.sort(key=lambda p: (len(p.parts), str(p)))
    return matches[0]


def unique_dst(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def plan(
    video_root: Path,
    min_age_minutes: float,
    segment_frames: int,
    quarantine_orphans: bool,
    filename_tz: ZoneInfo,
) -> tuple[list[Operation], dict[str, Any]]:
    legacy_videos, skipped_recent = find_legacy_videos(video_root, min_age_minutes)
    by_base: dict[str, list[LegacyVideo]] = {}
    for video in legacy_videos:
        by_base.setdefault(video.base, []).append(video)

    operations: list[Operation] = []
    stats: dict[str, Any] = {
        "video_root": str(video_root),
        "legacy_mp4_count": len(legacy_videos),
        "base_count": len(by_base),
        "skipped_recent_count": len(skipped_recent),
        "skipped_recent": [str(p) for p in skipped_recent[:50]],
        "bases": {},
    }

    for base, videos in sorted(by_base.items()):
        videos.sort(key=lambda v: (v.block, v.cam, str(v.path)))
        json_path = find_json_for_base(video_root, base, videos[0].path.parent)
        base_stats: dict[str, Any] = {
            "mp4_count": len(videos),
            "blocks": sorted({v.block for v in videos}),
            "json": str(json_path) if json_path else None,
            "status": "planned",
        }

        if json_path is None:
            base_stats["status"] = "orphan_no_json"
            if quarantine_orphans:
                for video in videos:
                    rel = video.path.relative_to(video_root)
                    dst = unique_dst(sidecar_dir(video_root, "legacy_unusable") / rel)
                    operations.append(
                        Operation("move", str(video.path), str(dst), f"orphan legacy MP4 has no {base}.json")
                    )
            else:
                base_stats["note"] = (
                    "Left in place. Pass --quarantine-orphans to move these out of VIDEO. "
                    "They cannot be converted for video-sync-nbu without a matching JSON."
                )
            stats["bases"][base] = base_stats
            continue

        if is_recent(json_path, min_age_minutes):
            base_stats["status"] = "skipped_recent_json"
            stats["bases"][base] = base_stats
            continue

        payload = load_json(json_path)
        total_frames = len(payload.get("real_times") or payload.get("timestamps") or [])
        max_block = max(v.block for v in videos)
        expected_blocks = max_block + 1
        base_stats["total_json_frames"] = total_frames
        base_stats["expected_blocks_from_mp4"] = expected_blocks

        if total_frames <= 0:
            base_stats["status"] = "bad_json_no_frames"
            stats["bases"][base] = base_stats
            continue

        block_to_seg: dict[int, str] = {}
        for block in sorted({v.block for v in videos}):
            start = block * segment_frames
            end = min((block + 1) * segment_frames, total_frames)
            if start >= total_frames:
                base_stats.setdefault("skipped_blocks", []).append(block)
                continue
            split_payload = slice_json(payload, start, end)
            seg_id = infer_segment_id(base, block, split_payload, filename_tz)
            block_to_seg[block] = seg_id
            date_dir = date_dir_for(json_path, seg_id)
            dst_json = video_root / date_dir / f"{seg_id}.json"
            operations.append(
                Operation(
                    "write_json",
                    str(json_path),
                    str(dst_json),
                    f"split {base}.json frames [{start}:{end}] for block {block}",
                )
            )

        for video in videos:
            seg_id = block_to_seg.get(video.block)
            if seg_id is None:
                continue
            date_dir = date_dir_for(video.path, seg_id)
            dst_mp4 = video_root / date_dir / f"{seg_id}.{video.cam}.mp4"
            operations.append(
                Operation(
                    "move",
                    str(video.path),
                    str(dst_mp4),
                    f"legacy block {video.block} -> segment {seg_id}",
                )
            )

        base_stats["planned_segments"] = block_to_seg
        stats["bases"][base] = base_stats

    return operations, stats


def apply_operations(video_root: Path, operations: list[Operation], keep_original_json: bool) -> None:
    write_ops = [op for op in operations if op.action == "write_json"]
    payload_cache: dict[Path, dict[str, Any]] = {}
    for op in write_ops:
        assert op.src and op.dst
        src = Path(op.src)
        dst = Path(op.dst)
        if keep_original_json and src == dst:
            raise RuntimeError(
                f"Cannot keep original JSON in place because split output would overwrite it: {src}"
            )
        if src not in payload_cache:
            payload_cache[src] = load_json(src)

    archived_json_sources: set[Path] = set()
    for op in write_ops:
        assert op.src and op.dst
        src = Path(op.src)
        dst = Path(op.dst)
        if src == dst and not keep_original_json and src not in archived_json_sources:
            rel = src.relative_to(video_root)
            archive_dst = unique_dst(sidecar_dir(video_root, "legacy_originals") / rel)
            archive_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(archive_dst))
            archived_json_sources.add(src)

    for op in operations:
        if op.action == "write_json":
            assert op.src and op.dst
            src = Path(op.src)
            dst = Path(op.dst)
            payload = payload_cache[src]

            match = re.search(r"frames \[(\d+):(\d+)\]", op.reason)
            if not match:
                raise RuntimeError(f"Could not parse frame slice from operation: {op}")
            start, end = int(match.group(1)), int(match.group(2))
            dump_json(dst, slice_json(payload, start, end))

        elif op.action == "move":
            assert op.src and op.dst
            src = Path(op.src)
            dst = Path(op.dst)
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise FileExistsError(f"Destination exists: {dst}")
            shutil.move(str(src), str(dst))

        else:
            raise ValueError(f"Unknown operation: {op.action}")

    if not keep_original_json:
        for src in sorted(payload_cache):
            if not src.exists():
                continue
            if src in archived_json_sources:
                # This source path is now a valid split JSON, not the original.
                continue
            # Only move the original JSON after all split JSONs were written.
            rel = src.relative_to(video_root)
            dst = unique_dst(sidecar_dir(video_root, "legacy_originals") / rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))


def cleanup_empty_dirs(video_root: Path) -> int:
    removed = 0
    for root, dirs, files in os.walk(video_root, topdown=False):
        path = Path(root)
        if path == video_root:
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()
            removed += 1
    return removed


def validate_patient_safety(video_root: Path, patient: str | None) -> None:
    if patient is None:
        return
    text = str(video_root)
    acceptable = [
        f"{patient}_Datafile",
        f"{patient}Datafile",
        f"/{patient}/",
    ]
    if not any(marker in text for marker in acceptable):
        raise RuntimeError(
            f"--patient {patient} does not appear to match --video-root {video_root}. "
            "Refusing to continue."
        )


def main() -> int:
    args = parse_args()
    video_root = args.video_root.expanduser().resolve()
    validate_patient_safety(video_root, args.patient)

    if not video_root.exists() or not video_root.is_dir():
        print(f"ERROR: VIDEO root does not exist or is not a directory: {video_root}", file=sys.stderr)
        return 2

    if args.delete_quarantine and not (args.execute and args.quarantine_orphans):
        print("ERROR: --delete-quarantine requires --execute --quarantine-orphans", file=sys.stderr)
        return 2

    try:
        filename_tz = ZoneInfo(args.filename_timezone)
    except Exception as exc:
        print(f"ERROR: invalid --filename-timezone {args.filename_timezone!r}: {exc}", file=sys.stderr)
        return 2

    operations, stats = plan(
        video_root=video_root,
        min_age_minutes=args.min_age_minutes,
        segment_frames=args.segment_frames,
        quarantine_orphans=args.quarantine_orphans,
        filename_tz=filename_tz,
    )

    manifest = args.manifest
    if manifest is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.execute:
            manifest = (
                sidecar_dir(video_root, "reformat_manifests")
                / f"reformat_legacy_emu_video_{stamp}.manifest.json"
            )
        else:
            manifest = Path("/tmp") / f"reformat_legacy_emu_video_{stamp}.manifest.json"

    manifest_payload = {
        "mode": "execute" if args.execute else "dry_run",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stats": stats,
        "operations": [asdict(op) for op in operations],
    }
    dump_json(manifest, manifest_payload)

    print(f"VIDEO root: {video_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"Legacy MP4s found: {stats['legacy_mp4_count']}")
    print(f"Bases found: {stats['base_count']}")
    print(f"Recent files skipped: {stats['skipped_recent_count']}")
    print(f"Planned operations: {len(operations)}")
    print(f"Manifest: {manifest}")

    if not args.execute:
        print("\nDry run only. Re-run with --execute after the copy is complete.")
        return 0

    apply_operations(video_root, operations, keep_original_json=args.keep_original_json)
    removed_dirs = cleanup_empty_dirs(video_root)
    print(f"Applied operations. Removed {removed_dirs} empty legacy directories.")

    quarantine = sidecar_dir(video_root, "legacy_unusable")
    if args.delete_quarantine and quarantine.exists():
        shutil.rmtree(quarantine)
        print(f"Deleted quarantine: {quarantine}")

    print("\nNext check:")
    print(f"  python /scratch/tahaismail424/video-sync-nbu/scripts/cli/cli_emu_time.py --help")
    print(
        "  or validate individual date folders with "
        "/scratch/tahaismail424/video-sync-nbu/scripts/validate/validate_video_dir.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
