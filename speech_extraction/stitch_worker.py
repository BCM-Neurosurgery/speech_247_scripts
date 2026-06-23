import os
import sys
import json
import traceback
from pathlib import Path
from glob import glob
from datetime import datetime

import pandas as pd

from pyNsXStitch.stitchers import StitchedNeVFile, StitchedNsXFile
from pyNsXStitch.helpers import find_nsx_in_range


def sorted_blackrock_files(pattern: str):
    files = glob(pattern)
    return sorted(files, key=lambda a: int(a[-7:-4]))


def safe_write_json(path: Path, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python stitch_interval_worker.py <manifest_csv> <row_idx> <write_root>",
            file=sys.stderr
        )
        sys.exit(2)

    manifest_csv = sys.argv[1]
    row_idx = int(sys.argv[2])
    write_root = Path(sys.argv[3])

    df = pd.read_csv(manifest_csv, parse_dates=["utc_ts", "utc_te"])

    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"row_idx {row_idx} out of range for manifest of length {len(df)}")

    row = df.iloc[row_idx]

    patient = row["patient"]
    toc = row["toc"]
    interval_id = row["interval_id"]
    br_ts = int(row["br_ts"])
    br_te = int(row["br_te"])
    utc_ts = row["utc_ts"]
    utc_te = row["utc_te"]

    data_root = Path("/mnt/datalake/data/emu")
    recording_dir = data_root / f"{patient}Datafile" / "DATA" / toc

    if not recording_dir.exists():
        raise FileNotFoundError(f"Recording directory does not exist: {recording_dir}")

    out_dir = write_root / patient / "vad_data" / interval_id
    neural_dir = out_dir / "neural"
    ensure_dir(neural_dir)

    metadata_path = out_dir / "metadata.json"
    error_path = out_dir / "stitch_error.txt"
    done_path = out_dir / "_SUCCESS"

    # Skip if already completed
    if done_path.exists():
        print(f"[SKIP] already completed: {interval_id}")
        return

    try:
        # discover raw files
        nsp1_ns5_files = sorted_blackrock_files(str(recording_dir / "NSP1*.ns5"))
        nsp2_ns5_files = sorted_blackrock_files(str(recording_dir / "NSP2*.ns5"))
        nsp1_ns3_files = sorted_blackrock_files(str(recording_dir / "NSP1*.ns3"))
        nsp1_nev_files = sorted_blackrock_files(str(recording_dir / "NSP1*.nev"))
        nsp2_nev_files = sorted_blackrock_files(str(recording_dir / "NSP2*.nev"))

        metadata = {
            "patient": patient,
            "toc": toc,
            "interval_id": interval_id,
            "utc_ts": str(utc_ts),
            "utc_te": str(utc_te),
            "br_ts": br_ts,
            "br_te": br_te,
            "duration_s": float((pd.Timestamp(utc_te) - pd.Timestamp(utc_ts)).total_seconds()),
            "recording_dir": str(recording_dir),
            "input_files": {
                "nsp1_ns5_count": len(nsp1_ns5_files),
                "nsp2_ns5_count": len(nsp2_ns5_files),
                "nsp1_ns3_count": len(nsp1_ns3_files),
                "nsp1_nev_count": len(nsp1_nev_files),
                "nsp2_nev_count": len(nsp2_nev_files),
            },
        }

        # select only files overlapping requested br range
        nsp1_ns5_to_stitch = find_nsx_in_range(nsp1_ns5_files, br_ts, br_te) if nsp1_ns5_files else []
        nsp2_ns5_to_stitch = find_nsx_in_range(nsp2_ns5_files, br_ts, br_te) if nsp2_ns5_files else []
        nsp1_ns3_to_stitch = find_nsx_in_range(nsp1_ns3_files, br_ts, br_te) if nsp1_ns3_files else []

        metadata["selected_files"] = {
            "nsp1_ns5": nsp1_ns5_to_stitch,
            "nsp2_ns5": nsp2_ns5_to_stitch,
            "nsp1_ns3": nsp1_ns3_to_stitch,
            "nsp1_nev": nsp1_nev_files,
            "nsp2_nev": nsp2_nev_files,
        }

        # write metadata early
        safe_write_json(metadata_path, metadata)

        # NSP-1 NS5
        if nsp1_ns5_to_stitch:
            out_path = neural_dir / f"{patient}_{interval_id}_NSP-1.ns5"
            ns5 = StitchedNsXFile(nsp1_ns5_to_stitch, start=br_ts, end=br_te, aggressive_concat=True)
            with open(out_path, "wb+") as f:
                ns5.write(f)
            print(f"[OK] wrote {out_path}")
        else:
            print("[WARN] no NSP-1 NS5 files overlapping interval")

        # NSP-2 NS5
        if nsp2_ns5_to_stitch:
            out_path = neural_dir / f"{patient}_{interval_id}_NSP-2.ns5"
            ns5 = StitchedNsXFile(nsp2_ns5_to_stitch, start=br_ts, end=br_te, aggressive_concat=True)
            with open(out_path, "wb+") as f:
                ns5.write(f)
            print(f"[OK] wrote {out_path}")
        else:
            print("[WARN] no NSP-2 NS5 files overlapping interval")

        # NSP-1 NS3
        if nsp1_ns3_to_stitch:
            out_path = neural_dir / f"{patient}_{interval_id}_NSP-1.ns3"
            ns3 = StitchedNsXFile(nsp1_ns3_to_stitch, start=br_ts, end=br_te, aggressive_concat=True)
            with open(out_path, "wb+") as f:
                ns3.write(f)
            print(f"[OK] wrote {out_path}")
        else:
            print("[WARN] no NSP-1 NS3 files overlapping interval")

        # NSP-1 NEV
        if nsp1_nev_files:
            out_path = neural_dir / f"{patient}_{interval_id}_NSP-1.nev"
            nev = StitchedNeVFile(nsp1_nev_files, start=br_ts, end=br_te)
            with open(out_path, "wb") as f:
                nev.write(f)
            print(f"[OK] wrote {out_path}")
        else:
            print("[WARN] no NSP-1 NEV files found")

        # NSP-2 NEV
        if nsp2_nev_files:
            out_path = neural_dir / f"{patient}_{interval_id}_NSP-2.nev"
            nev = StitchedNeVFile(nsp2_nev_files, start=br_ts, end=br_te)
            with open(out_path, "wb") as f:
                nev.write(f)
            print(f"[OK] wrote {out_path}")
        else:
            print("[WARN] no NSP-2 NEV files found")

        # refresh metadata with output file list
        metadata["output_files"] = sorted([str(p) for p in neural_dir.glob("*")])
        safe_write_json(metadata_path, metadata)

        done_path.write_text("ok\n")
        if error_path.exists():
            error_path.unlink()

        print(f"[DONE] {interval_id}")

    except Exception as e:
        with open(error_path, "w") as f:
            f.write(f"FAILED interval_id={interval_id}\n")
            f.write(f"{type(e).__name__}: {e}\n\n")
            f.write(traceback.format_exc())
        print(f"[FAIL] {interval_id}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()