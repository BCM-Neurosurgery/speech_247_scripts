"""
Spike detection worker for one stitched interval directory.

Usage:
    python spike_thresholding_worker.py <interval_dir> [options]

Output is written to:
    <interval_dir>/neural/spike_thresholding/
        binary_spiketrain.npy   shape (n_channels, n_bins) uint8 at out_fs
        spike_stats.json        thresholds, per-channel rates, etc.
        channel_names.json      ordered list of channel names
        _SUCCESS                written last on clean exit

Errors are written to:
    <interval_dir>/neural/spike_thresholding/error.txt
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass_filter(x, fs, lo=300, hi=6000, order=4, notch_hz=None, q=15):
    ny = 0.5 * fs
    xf = x
    if notch_hz is not None:
        b_notch, a_notch = iirnotch(notch_hz / ny, q)
        xf = filtfilt(b_notch, a_notch, x, axis=-1)
    b, a = butter(order, [lo / ny, hi / ny], btype="band")
    return filtfilt(b, a, xf, axis=-1)


def robust_sigma(x, eps=1e-9):
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True)
    return 1.4826 * mad + eps


def _detect_peaks_channel(xc, thr, min_dist):
    peaks, _ = find_peaks(-xc, height=thr, distance=min_dist)
    return peaks.astype(np.int64)


def _avg_rate(xf, fs, k, sigmas, refractory_s):
    min_dist = max(1, int(refractory_s * fs))
    n_ch, n_samp = xf.shape
    total = sum(
        len(_detect_peaks_channel(xf[c], k * float(sigmas[c, 0]), min_dist))
        for c in range(n_ch)
    )
    return total / (n_samp / fs * n_ch)


def _solve_k(xf, fs, target_hz, refractory_s, k_low=1.5, k_high=10.0, tol_hz=0.5, max_iter=25):
    sigmas = robust_sigma(xf)
    r_low = _avg_rate(xf, fs, k_low, sigmas, refractory_s)
    r_high = _avg_rate(xf, fs, k_high, sigmas, refractory_s)

    for _ in range(6):
        if r_low >= target_hz:
            break
        k_low *= 0.7
        r_low = _avg_rate(xf, fs, k_low, sigmas, refractory_s)
    for _ in range(6):
        if r_high <= target_hz:
            break
        k_high *= 1.5
        r_high = _avg_rate(xf, fs, k_high, sigmas, refractory_s)

    for _ in range(max_iter):
        k_mid = 0.5 * (k_low + k_high)
        r_mid = _avg_rate(xf, fs, k_mid, sigmas, refractory_s)
        if abs(r_mid - target_hz) <= tol_hz:
            return k_mid, sigmas
        if r_mid > target_hz:
            k_low = k_mid
        else:
            k_high = k_mid
    return 0.5 * (k_low + k_high), sigmas


def detect_and_bin_spikes(
    x_raw,
    fs=30000,
    out_fs=1000,
    refractory_ms=1.0,
    notch_hz=None,
    band_lo=300,
    band_hi=6000,
    target_avg_hz=20.0,
    tol_hz=0.5,
    k_low=1.5,
    k_high=10.0,
    solve_subset_sec=60,
):
    n_ch, n_samp = x_raw.shape
    refractory_s = refractory_ms / 1000.0

    xf_full = bandpass_filter(x_raw, fs, band_lo, band_hi, notch_hz=notch_hz)

    subset = xf_full[:, : min(int(solve_subset_sec * fs), n_samp)]
    k, sigmas = _solve_k(subset, fs, target_avg_hz, refractory_s, k_low, k_high, tol_hz)

    min_dist = max(1, int(refractory_s * fs))
    spike_indices = [
        _detect_peaks_channel(xf_full[c], k * float(sigmas[c, 0]), min_dist)
        for c in range(n_ch)
    ]

    duration_s = n_samp / fs
    n_bins = int(np.ceil(duration_s * out_fs))
    spike_bin = np.zeros((n_ch, n_bins), dtype=np.uint8)
    for c, idx in enumerate(spike_indices):
        if idx.size:
            bins = np.unique((idx * out_fs) // fs)
            spike_bin[c, bins[bins < n_bins]] = 1

    per_ch_counts = np.array([len(i) for i in spike_indices], dtype=float)
    per_ch_rates = per_ch_counts / duration_s

    return spike_bin, {
        "k": float(k),
        "sigmas_uV": sigmas.squeeze(-1),
        "per_channel_rates_Hz": per_ch_rates,
        "achieved_avg_rate_Hz": float(per_ch_rates.mean()),
        "n_bins": n_bins,
        "out_fs": out_fs,
        "fs": fs,
        "duration_s": float(duration_s),
        "n_channels": int(n_ch),
    }


# ---------------------------------------------------------------------------
# Neural data loading via Neo
# ---------------------------------------------------------------------------

def _pick_neural_signal(analogsignals, target_fs):
    """
    Select the AnalogSignal that represents micro-electrode neural data:
    - sampling rate closest to target_fs (30 kHz)
    - units of uV (not mV / behavioral)
    - most channels among candidates
    """
    candidates = []
    for sig in analogsignals:
        fs_sig = float(sig.sampling_rate.rescale("Hz").magnitude)
        units = str(sig.units).strip()
        n_ch = sig.shape[1] if len(sig.shape) > 1 else 1
        candidates.append((sig, fs_sig, units, n_ch))

    # Filter to signals near target_fs
    near = [(s, f, u, n) for s, f, u, n in candidates if abs(f - target_fs) / target_fs < 0.05]
    if not near:
        # Fall back to the one with the highest sampling rate
        near = sorted(candidates, key=lambda x: -x[1])[:1]

    # Among near-target-fs candidates, prefer uV over mV, then most channels
    uv_cands = [(s, f, u, n) for s, f, u, n in near if "uV" in u]
    pool = uv_cands if uv_cands else near
    return max(pool, key=lambda x: x[3])[0]


def load_neural_data(ns5_path: Path, target_fs: int = 30000):
    """
    Load neural micro-electrode data from a stitched NSP ns5 file.
    Concatenates across multiple blocks/segments with zero-padding for gaps.
    Returns (data_uV: ndarray (n_ch, n_samp), fs: float, channel_names: list[str])
    """
    from neo.io import BlackrockIO

    reader = BlackrockIO(str(ns5_path))
    n_blocks = reader.header["nb_block"]
    n_segs = reader.header["nb_segment"]

    pieces = []
    fs_out = None
    channel_names = None
    prev_end_sample = None

    for bi in range(n_blocks):
        blk = reader.read_block(block_index=bi, lazy=True)
        for si in range(n_segs[bi]):
            seg = blk.segments[si]
            sig = _pick_neural_signal(seg.analogsignals, target_fs)
            fs_sig = float(sig.sampling_rate.rescale("Hz").magnitude)

            if fs_out is None:
                fs_out = fs_sig
            if channel_names is None:
                channel_names = [str(ch) for ch in sig.array_annotations.get("channel_names", range(sig.shape[1]))]

            loaded = sig.load()
            arr = np.asarray(loaded).T.astype(np.float32)  # (n_ch, n_samp)

            if prev_end_sample is not None:
                seg_start_sample = int(round(float(sig.t_start.rescale("s").magnitude) * fs_out))
                gap = seg_start_sample - prev_end_sample
                if gap > 0:
                    pieces.append(np.zeros((arr.shape[0], gap), dtype=np.float32))

            pieces.append(arr)
            prev_end_sample = (int(round(float(sig.t_start.rescale("s").magnitude) * fs_out))
                               + arr.shape[1])

    data = np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
    return data, fs_out, channel_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interval_dir", type=Path)
    parser.add_argument("--nsp", type=int, default=2,
                        help="NSP number to use (default: 2, which has micro-electrode channels)")
    parser.add_argument("--target-hz", type=float, default=20.0)
    parser.add_argument("--notch", type=float, default=60.0)
    parser.add_argument("--out-fs", type=int, default=1000)
    parser.add_argument("--solve-subset-sec", type=float, default=60.0)
    args = parser.parse_args()

    interval_dir = args.interval_dir.resolve()
    out_dir = interval_dir / "neural" / "spike_thresholding"
    out_dir.mkdir(parents=True, exist_ok=True)
    success_path = out_dir / "_SUCCESS"
    error_path = out_dir / "error.txt"

    if success_path.exists():
        print(f"already done: {interval_dir}", flush=True)
        sys.exit(0)

    try:
        # Find the NS5 file
        ns5_candidates = sorted((interval_dir / "neural").glob(f"*_NSP-{args.nsp}.ns5"))
        if not ns5_candidates:
            raise FileNotFoundError(f"No NSP-{args.nsp} ns5 file in {interval_dir}/neural/")
        ns5_path = ns5_candidates[0]
        print(f"loading {ns5_path.name}", flush=True)

        data_uV, fs, channel_names = load_neural_data(ns5_path)
        print(f"  shape={data_uV.shape} fs={fs:.0f}Hz  channels={len(channel_names)}", flush=True)

        print(f"  running spike detection (target={args.target_hz} Hz, notch={args.notch} Hz)", flush=True)
        spike_bin, info = detect_and_bin_spikes(
            data_uV,
            fs=int(round(fs)),
            out_fs=args.out_fs,
            notch_hz=args.notch,
            target_avg_hz=args.target_hz,
            solve_subset_sec=args.solve_subset_sec,
        )
        print(f"  achieved avg rate: {info['achieved_avg_rate_Hz']:.2f} Hz  k={info['k']:.3f}", flush=True)

        np.save(str(out_dir / "binary_spiketrain.npy"), spike_bin)
        with open(out_dir / "channel_names.json", "w") as f:
            json.dump(channel_names, f, indent=2)
        stats = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
        stats["ns5_path"] = str(ns5_path)
        stats["nsp"] = args.nsp
        stats["notch_hz"] = args.notch
        stats["target_avg_hz"] = args.target_hz
        with open(out_dir / "spike_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        success_path.write_text("ok\n")
        print("done", flush=True)

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        print(f"FAILED:\n{tb}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
