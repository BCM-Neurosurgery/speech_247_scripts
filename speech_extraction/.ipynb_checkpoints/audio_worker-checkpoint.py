import os
import sys
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from neo.io import BlackrockIO


ORIGINAL_FS = 30000
TARGET_FS = 16000


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python extract_interval_audio.py <manifest_csv> <row_idx> <audio_chan>",
            file=sys.stderr
        )
        sys.exit(2)

    manifest_csv = sys.argv[1]
    row_idx = int(sys.argv[2])
    audio_chan = int(sys.argv[3])

    df = pd.read_csv(manifest_csv, parse_dates=["utc_ts", "utc_te"])

    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"row_idx {row_idx} out of range")

    row = df.iloc[row_idx]

    patient = row["patient"]
    interval_id = row["interval_id"]

    write_root = Path("/mnt/labworlds/Hayden/Hayden_Lab/speech_247/vad_new")
    interval_dir = write_root / patient / "vad_data" / interval_id
    neural_dir = interval_dir / "neural"
    transcription_dir = interval_dir / "transcription"
    whisperx_dir = transcription_dir / "whisperx"

    transcription_dir.mkdir(parents=True, exist_ok=True)
    whisperx_dir.mkdir(parents=True, exist_ok=True)

    audio_path = transcription_dir / "audio.wav"
    error_path = transcription_dir / "transcription_error.txt"

    ns5_path = neural_dir / f"{patient}_{interval_id}_NSP-1.ns5"
    if not ns5_path.exists():
        raise FileNotFoundError(f"Missing NSP-1 ns5 file: {ns5_path}")

    try:
        reader = BlackrockIO(str(ns5_path))
        n_blocks = reader.header["nb_block"]
        n_segments = reader.header["nb_segment"]

        audio_chunks = []

        for block_idx in range(n_blocks):
            block = reader.read_block(block_index=block_idx, lazy=True)

            for segment_idx in range(n_segments[block_idx]):
                segment = block.segments[segment_idx]

                analog_signal = None
                for signal in segment.analogsignals:
                    # choose non-nsx3 continuous stream
                    if signal.name.item() != "nsx3":
                        analog_signal = signal
                        break

                if analog_signal is None:
                    continue

                data = analog_signal.load()
                audio = np.asarray(data[:, audio_chan]).squeeze()

                if audio.ndim != 1:
                    raise ValueError(f"Audio channel extraction did not produce 1D array for {ns5_path}")

                audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError(f"No audio chunks found in {ns5_path}")

        audio = np.concatenate(audio_chunks).astype(np.float32)

        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs

        audio_t = torch.from_numpy(audio)
        audio_rs = torchaudio.functional.resample(audio_t, ORIGINAL_FS, TARGET_FS).numpy()

        max_abs_rs = np.max(np.abs(audio_rs))
        if max_abs_rs > 0:
            audio_rs = audio_rs / max_abs_rs

        sf.write(audio_path, audio_rs, TARGET_FS, subtype="PCM_16")

        meta = {
            "patient": patient,
            "interval_id": interval_id,
            "source_ns5": str(ns5_path),
            "audio_path": str(audio_path),
            "audio_chan": audio_chan,
            "original_fs": ORIGINAL_FS,
            "target_fs": TARGET_FS,
            "n_samples": int(audio_rs.shape[0]),
            "duration_s": float(audio_rs.shape[0] / TARGET_FS),
        }

        with open(transcription_dir / "audio_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if error_path.exists():
            error_path.unlink()

        print(f"[OK] wrote {audio_path}")

    except Exception as e:
        with open(error_path, "w") as f:
            f.write(f"{type(e).__name__}: {e}\n\n")
            f.write(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()