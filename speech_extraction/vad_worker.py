import os
import sys
import traceback
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torchaudio
from neo.io import BlackrockIO
from silero_vad import load_silero_vad, get_speech_timestamps
from pyNsXStitch.helpers import get_nsx_start_timestamp

ORIGINAL_FS = 30_000
TARGET_FS = 16_000
AUDIO_CHAN = 2


def run_vad_one_file(ns5_file: str, out_root: str) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_silero_vad().to(device)

    ns5_path = Path(ns5_file)
    patient = ns5_path.parts[5][:3] if len(ns5_path.parts) > 4 else ns5_path.name[:3]
    patient_out = Path(out_root) / patient
    patient_out.mkdir(parents=True, exist_ok=True)

    chunk_name = ns5_path.stem
    toc = '-'.join(chunk_name.split('-')[1:-1]) if '-' in chunk_name else chunk_name

    start_br = get_nsx_start_timestamp(str(ns5_path))
    br_file = BlackrockIO(str(ns5_path))
    br_sync_utc = br_file.raw_annotations['blocks'][0]['rec_datetime']

    n_blocks = br_file.header['nb_block']
    n_segments = br_file.header['nb_segment']

    intervals = []

    for block_idx in range(n_blocks):
        utc_stamp = br_file.raw_annotations['blocks'][block_idx]['rec_datetime']
        block_data = br_file.read_block(block_index=block_idx, lazy=True)

        start_stamp = 0.0

        for segment_idx in range(n_segments[block_idx]):
            segment = block_data.segments[segment_idx]

            analog_signal = None
            for signal in segment.analogsignals:
                name = signal.name
                if not isinstance(name, str):
                    name = name.item()
                if name != 'nsx3' and signal.shape[1] < 30:
                    analog_signal = signal
                    break

            if analog_signal is None:
                continue

            if segment_idx == 0:
                start_stamp = float(analog_signal.t_start)

            offset = float(analog_signal.t_start) - start_stamp

            data = analog_signal.load()
            audio = data[:, AUDIO_CHAN]
            audio = torch.as_tensor(audio).squeeze().float()

            max_abs = torch.max(torch.abs(audio))
            if max_abs > 0:
                audio = audio / max_abs
            else:
                # silent segment
                continue

            audio = torchaudio.functional.resample(audio, ORIGINAL_FS, TARGET_FS)
            audio = audio.to(device)

            ts = get_speech_timestamps(audio, model, return_seconds=True)

            utc_offset = utc_stamp + timedelta(seconds=offset)
            br_offset = start_br + int(np.round((utc_stamp - br_sync_utc).total_seconds() * ORIGINAL_FS))

            for interval in ts:
                intervals.append({
                    "patient": patient,
                    "toc": toc,
                    "chunk": chunk_name,
                    "block": block_idx,
                    "segment": segment_idx,
                    "segment_ts_s": interval["start"],
                    "segment_te_s": interval["end"],
                    "utc_ts": utc_offset + timedelta(seconds=interval["start"]),
                    "utc_te": utc_offset + timedelta(seconds=interval["end"]),
                    "br_ts": br_offset + int(np.round(ORIGINAL_FS * interval["start"])),
                    "br_te": br_offset + int(np.round(ORIGINAL_FS * interval["end"]))
                })

    return pd.DataFrame(intervals)


def main():
    if len(sys.argv) != 3:
        print("Usage: python vad_worker.py <ns5_file> <out_root>", file=sys.stderr)
        sys.exit(2)

    ns5_file = sys.argv[1]
    out_root = sys.argv[2]

    ns5_path = Path(ns5_file)
    patient = ns5_path.parts[5][:3] if len(ns5_path.parts) > 4 else ns5_path.name[:3]
    patient_out = Path(out_root) / patient
    patient_out.mkdir(parents=True, exist_ok=True)

    chunk_name = ns5_path.stem
    csv_path = patient_out / "vad_outs" / f"{chunk_name}_vad.csv"
    err_path = patient_out / "vad_outs" / f"{chunk_name}_vad.error.txt"

    try:
        df = run_vad_one_file(ns5_file, out_root)
        df.to_csv(csv_path, index=False)

        # remove stale error file from prior failed run
        if err_path.exists():
            err_path.unlink()

        print(f"[OK] wrote {csv_path}")
    except Exception as e:
        with open(err_path, "w") as f:
            f.write(f"FAILED: {ns5_file}\n")
            f.write(f"{type(e).__name__}: {e}\n\n")
            f.write(traceback.format_exc())
        print(f"[FAIL] {ns5_file}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()