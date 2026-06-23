"""
Quality assessment worker for a single interval.

Usage:
    python quality_worker.py <vad_data_root> <interval_id>

Reads:
    {interval_dir}/transcription/audio.wav
    {interval_dir}/transcription/whisperx/audio.json

Computes per-segment:
    - ctc_loss:          negative mean CTC loss via wav2vec2-large-960h-lv60-self (GPU)
    - spectral_entropy:  mean spectral entropy via librosa (CPU)
    - dnsmos_sig/bak/ovrl: DNSMOS P.835 via ONNX (GPU/CPU)

Writes:
    {interval_dir}/quality/segment_quality.csv
    {interval_dir}/quality/_SUCCESS  (or _ERROR on failure)
"""

import json
import os
import sys
import traceback
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.stats
import soundfile as sf
import torch

FS = 16_000
DNSMOS_INPUT_LENGTH_S = 9.01  # model expects this many seconds; shorter clips are padded
DNSMOS_INPUT_SAMPLES = int(DNSMOS_INPUT_LENGTH_S * FS)

# Configurable ONNX model path — set DNSMOS_ONNX_PATH env var or update this list
_DNSMOS_ONNX_CANDIDATES = [
    os.environ.get("DNSMOS_ONNX_PATH", ""),
    "/scratch/tahaismail424/speech_247/models/dnsmos/sig_bak_ovr.onnx",
    "/scratch/tahaismail424/hf/dnsmos/sig_bak_ovr.onnx",
    "/scratch/ti12/speech_247/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
]


# ── Audio quality metrics ───────────────────────────────────────────────────────

def spectral_entropy(audio: np.ndarray, sr: int = FS) -> float:
    S = np.abs(librosa.stft(audio)) ** 2
    col_sum = S.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1e-10
    psd = S / col_sum
    entropy = scipy.stats.entropy(psd, base=2, axis=0)
    return float(np.mean(entropy))


def ctc_loss_score(
    audio: np.ndarray,
    transcript: str,
    processor,
    model,
    device: torch.device,
) -> float:
    transcript = transcript.strip().upper()
    if not transcript:
        return float("nan")
    inputs = processor(audio, sampling_rate=FS, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    labels = processor.tokenizer(transcript, return_tensors="pt").input_ids.to(device)
    input_lengths = torch.tensor([logits.shape[1]])
    target_lengths = torch.tensor([labels.shape[1]])
    ctc = torch.nn.CTCLoss(
        blank=processor.tokenizer.pad_token_id,
        reduction="mean",
        zero_infinity=True,
    ).to(device)
    loss = ctc(log_probs.transpose(0, 1), labels, input_lengths, target_lengths)
    return float(-loss.item())


def dnsmos_score(audio: np.ndarray, sess) -> tuple[float, float, float]:
    """Return (SIG, BAK, OVRL) from the DNSMOS ONNX session."""
    audio = audio.astype(np.float32)
    if len(audio) < DNSMOS_INPUT_SAMPLES:
        audio = np.pad(audio, (0, DNSMOS_INPUT_SAMPLES - len(audio)))
    else:
        audio = audio[:DNSMOS_INPUT_SAMPLES]
    audio_input = audio[np.newaxis, :]  # (1, T)
    ort_inputs = {sess.get_inputs()[0].name: audio_input}
    out = sess.run(None, ort_inputs)[0][0]  # (3,)
    return float(out[0]), float(out[1]), float(out[2])


# ── Model loaders ───────────────────────────────────────────────────────────────

def load_wav2vec2(device: torch.device):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    hf_home = os.environ.get("HF_HOME", "/scratch/tahaismail424/hf")
    model_id = "facebook/wav2vec2-large-960h-lv60-self"
    processor = Wav2Vec2Processor.from_pretrained(
        model_id, cache_dir=hf_home, local_files_only=True
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id, cache_dir=hf_home, local_files_only=True
    ).to(device)
    model.eval()
    return processor, model


def load_dnsmos():
    import onnxruntime as ort

    onnx_path = None
    for candidate in _DNSMOS_ONNX_CANDIDATES:
        if candidate and Path(candidate).exists():
            onnx_path = candidate
            break

    if onnx_path is None:
        raise FileNotFoundError(
            f"DNSMOS ONNX model not found. Set DNSMOS_ONNX_PATH or place model at "
            f"{_DNSMOS_ONNX_CANDIDATES[1]}"
        )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[DNSMOS] loaded from {onnx_path}")
    return sess


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python quality_worker.py <vad_data_root> <interval_id>",
            file=sys.stderr,
        )
        sys.exit(2)

    vad_data_root = Path(sys.argv[1])
    interval_id = sys.argv[2]
    interval_dir = vad_data_root / interval_id
    quality_dir = interval_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    success_path = quality_dir / "_SUCCESS"
    error_path = quality_dir / "_ERROR"
    success_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)

    try:
        audio_path = interval_dir / "transcription" / "audio.wav"
        tx_path = interval_dir / "transcription" / "whisperx" / "audio.json"

        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio: {audio_path}")
        if not tx_path.exists():
            raise FileNotFoundError(f"Missing transcript: {tx_path}")

        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != FS:
            raise ValueError(f"Expected {FS} Hz audio, got {sr} Hz")

        with open(tx_path) as f:
            wdata = json.load(f)

        segments = wdata.get("segments", [])
        if not segments:
            print(f"[WARN] No segments in {tx_path} — writing empty output")
            pd.DataFrame(
                columns=[
                    "interval_id", "segment_idx", "segment_start_s", "segment_end_s",
                    "segment_text", "avg_word_score", "ctc_loss",
                    "spectral_entropy", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
                ]
            ).to_csv(quality_dir / "segment_quality.csv", index=False)
            success_path.write_text("ok")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        print("Loading wav2vec2...")
        processor, w2v_model = load_wav2vec2(device)

        print("Loading DNSMOS...")
        try:
            dnsmos_sess = load_dnsmos()
            has_dnsmos = True
        except Exception as e:
            print(f"[WARN] DNSMOS unavailable: {e}", file=sys.stderr)
            has_dnsmos = False

        rows = []
        for seg_idx, seg in enumerate(segments):
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            seg_text = str(seg.get("text", "")).strip()
            word_scores = [w.get("score", float("nan")) for w in seg.get("words", [])]
            avg_word_score = float(np.nanmean(word_scores)) if word_scores else float("nan")

            start_sample = int(round(seg_start * FS))
            end_sample = int(round(seg_end * FS))
            seg_audio = audio[start_sample:end_sample]

            row = {
                "interval_id": interval_id,
                "segment_idx": seg_idx,
                "segment_start_s": seg_start,
                "segment_end_s": seg_end,
                "segment_text": seg_text,
                "avg_word_score": avg_word_score,
                "ctc_loss": float("nan"),
                "spectral_entropy": float("nan"),
                "dnsmos_sig": float("nan"),
                "dnsmos_bak": float("nan"),
                "dnsmos_ovrl": float("nan"),
            }

            if len(seg_audio) < 160:  # < 10ms — skip scoring
                rows.append(row)
                continue

            try:
                row["ctc_loss"] = ctc_loss_score(seg_audio, seg_text, processor, w2v_model, device)
            except Exception as e:
                print(f"[WARN] CTC failed seg {seg_idx}: {e}", file=sys.stderr)

            try:
                row["spectral_entropy"] = spectral_entropy(seg_audio)
            except Exception as e:
                print(f"[WARN] entropy failed seg {seg_idx}: {e}", file=sys.stderr)

            if has_dnsmos:
                try:
                    sig, bak, ovrl = dnsmos_score(seg_audio, dnsmos_sess)
                    row["dnsmos_sig"] = sig
                    row["dnsmos_bak"] = bak
                    row["dnsmos_ovrl"] = ovrl
                except Exception as e:
                    print(f"[WARN] DNSMOS failed seg {seg_idx}: {e}", file=sys.stderr)

            rows.append(row)
            if seg_idx % 20 == 0:
                print(f"  seg {seg_idx + 1}/{len(segments)}")

        out_df = pd.DataFrame(rows)
        out_df.to_csv(quality_dir / "segment_quality.csv", index=False)
        success_path.write_text("ok")
        print(f"[OK] {len(rows)} segments → {quality_dir / 'segment_quality.csv'}")

    except Exception as e:
        error_path.write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
