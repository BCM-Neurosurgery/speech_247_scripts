# speech_247_scripts

Analysis code for **"Estimation of neuronal tuning for word meaning from passively recorded naturalistic speech"** (Ismail et al., _Nature_, in submission).

This repository contains all notebooks and scripts used to process 24/7 EMU recordings, build encoding and decoding models, benchmark against ground-truth datasets, and generate the paper's six main figures.

---

## Overview

The pipeline processes incidental speech continuously recorded from 21 epilepsy patients in the EMU (>800 hours, ~5.3 million words). It covers:

1. **Voice activity detection & speech extraction** from raw Blackrock NS5 files
2. **WhisperX transcription** and **quality filtration**
3. **Video-assisted speaker diarization** via LR-ASD
4. **Spike auto-thresholding** from microwire recordings
5. **GPT-2 semantic embedding** generation
6. **Encoding models** (Poisson GLM with L2 regularization; pseudo-R²)
7. **Decoding models** (XGBoost semantic category classifier)
8. **Controlled-task comparisons** (conversation + podcast ground-truth datasets)
9. **Decimation analysis** (model performance vs. dataset size)
10. **Functional drift analysis** (cross-day generalization)
11. **Figure generation** for all main figures

---

## Table of Contents

- [Top-Level Scripts](#top-level-scripts)
- [speech\_extraction/](#speech_extraction)
- [speech\_filtration/](#speech_filtration)
- [video\_processing/](#video_processing)
- [signal\_processing/](#signal_processing)
- [generate\_embeddings/](#generate_embeddings)
- [standard\_encoding\_analysis/](#standard_encoding_analysis)
- [standard\_decoding\_analysis/](#standard_decoding_analysis)
- [convo\_behav\_comparison/](#convo_behav_comparison)
- [decimation\_analysis/](#decimation_analysis)
- [functional\_drift/](#functional_drift)
- [controlled\_comparison/](#controlled_comparison)
  - [convo\_encoding\_comparison/](#convo_encoding_comparison)
  - [convo\_scat\_comparison/](#convo_scat_comparison)
  - [podcast\_encoding\_comparison/](#podcast_encoding_comparison)
  - [podcast\_scat\_comparison/](#podcast_scat_comparison)
- [figure\_generation/](#figure_generation)
- [Potentially Stale Notebooks](#potentially-stale-notebooks)

---

## Top-Level Scripts

| File | Description | Paper Section |
|------|-------------|---------------|
| `create_whisper_jobs.sh` | Scans `vad_out/{patient}` for untranscribed `.wav` files and writes a job list (`audio_jobs.txt`) for batch WhisperX transcription. Legacy variant of the SLURM notebook-based transcription pipeline. | Methods §Transcript Assembly |
| `execute_whisper_jobs.sh` | Reads `audio_jobs.txt` and runs WhisperX (large-v2 with wav2vec2 alignment and diarization) in parallel using GNU Parallel (6 jobs). Legacy companion to `create_whisper_jobs.sh`. | Methods §Transcript Assembly |

> **Note:** These shell scripts reflect an earlier pipeline iteration targeting the `vad_out` directory layout. The current SLURM-orchestrated workflow runs from within the notebooks in `speech_extraction/` and `generate_embeddings/`.

---

## speech\_extraction/

Identifies speech intervals in continuous recordings, runs WhisperX transcription, and assembles per-patient transcript tables.

| File | Description | Paper Section |
|------|-------------|---------------|
| `speech_extraction.ipynb` | **Main VAD orchestrator.** Loads raw NS5 neural recordings (30 kHz), resamples audio to 16 kHz, and runs Silero VAD to detect speech intervals. Produces per-interval audio files and a merged-interval manifest CSV. | Methods §Voice Activity Detection |
| `identify_segments_packetgaps.ipynb` | Scans NSP2 NS5 files to identify packet gaps and recording discontinuities before stitching. Used as a data-quality pre-check for new patients. | Methods §Electrophysiology Recording |
| `assemble_transcripts.ipynb` | Loads per-interval WhisperX JSON outputs and assembles them into a single `{patient}_transcripts.csv` (one row per word) with context strings for GPT-2. | Methods §Transcript Assembly |
| `sample_alignment_checks.ipynb` | Spot-checks audio/video/transcript alignment for a given patient by sampling short clips and displaying them side-by-side with LR-ASD video and transcript. QC utility, not used in final analysis. | Methods §Active Speaker Detection (QC) |
| `vad_worker.py` | SLURM worker called by `speech_extraction.ipynb`; processes a single VAD interval (audio extraction, stitching, transcription job prep). | Methods §Voice Activity Detection |
| `stitch_worker.py` | Worker that stitches multiple NS5 segments spanning a VAD interval into a continuous NS5 file for downstream spike thresholding. | Methods §Electrophysiology Recording |
| `audio_worker.py` | Worker that extracts and resamples audio from a Blackrock NS5 recording segment. | Methods §Voice Activity Detection |

> ⚠️ **Possibly stale:** `identify_channel_discreps.ipynb` and `identify_clock_resets.ipynb` are both empty notebooks. Flagged for manual review.

---

## speech\_filtration/

Computes per-sentence quality scores and filters low-quality transcriptions before downstream modeling.

| File | Description | Paper Section |
|------|-------------|---------------|
| `speech_filtration.ipynb` | **Main quality filtration orchestrator.** For each patient, submits SLURM jobs via `quality_worker.py` to compute CTC log-likelihood (wav2vec2), WhisperX alignment score, spectral entropy, and DNSMOS quality score per sentence. Filters sentences outside 0.75×IQR thresholds, producing `{patient}_word_df_filtered.csv`. | Methods §Quality Metrics; Results §Dataset |
| `quality_worker.py` | SLURM worker that computes the four quality metrics for a single VAD interval and writes results to `quality/`. | Methods §Quality Metrics |

---

## video\_processing/

Syncs video to neural timelines and runs LR-ASD active speaker detection to identify patient-spoken segments.

| File | Description | Paper Section |
|------|-------------|---------------|
| `run_video_sync_patient.ipynb` | **Per-interval video sync orchestrator.** For a chosen patient and camera serial, submits one SLURM job per VAD interval to align video timestamps with the neural clock via `video_sync_worker.py`. | Methods §Active Speaker Detection |
| `run_lrasd_patient.ipynb` | **LR-ASD orchestrator.** For a chosen patient and camera serial, submits one SLURM job per synced interval to run the LR-ASD active speaker detection model (`lrasd_interval_worker.py`). Optionally filters to intervals with good sync quality. | Methods §Active Speaker Detection |
| `add_video_diarization_df.ipynb` | Reads per-frame LR-ASD speaker scores and annotates the transcript word/segment DataFrame with `patient_speaking` flags. Produces `{patient}_transcripts_annotated.csv` used by patient-speech encoding/decoding analyses. Generates subtitled validation clips for review. | Methods §Active Speaker Detection; Results §Dataset §diarization |
| `stitch_videos_patient_speech.ipynb` | Concatenates patient-speech video clips across a recording stay for visual review and presentation. Not used in quantitative analysis. | — |
| `sync_video_patient.ipynb` | Earlier (pre-SLURM) version of video synchronization using `batch_sync_patient`. Superseded by `run_video_sync_patient.ipynb`. | — |
| `video_sync_worker.py` | SLURM worker that calls the `video-sync-nbu` package to align a single interval's video stream with neural timestamps. | Methods §Active Speaker Detection |
| `lrasd_interval_worker.py` | SLURM worker that runs LR-ASD on a single synced interval video. | Methods §Active Speaker Detection |
| `reformat_legacy_emu_video.py` | Reformats legacy EMU video files from older patients into the format expected by the current sync pipeline. | — |
| `add_nsp1_data.ipynb` | Adds NSP1 LFP/analog data to the patient data structure for a batch of patients. Targets the older `vad_out` layout. | — |
| `add_nsp1_data_all_patients.ipynb` | Single-patient version of `add_nsp1_data.ipynb`. Also targets the `vad_out` layout. | — |

> ⚠️ **Possibly stale:** `add_nsp1_data.ipynb` and `add_nsp1_data_all_patients.ipynb` reference the legacy `vad_out` directory and may not be compatible with the current `vad_new` pipeline. `sync_video_patient.ipynb` is a superseded version. Flagged for manual review.

---

## signal\_processing/

Applies spike auto-thresholding to bandpass-filtered microwire recordings and assembles word-aligned neural features.

| File | Description | Paper Section |
|------|-------------|---------------|
| `spike_thresholding.ipynb` | **Spike thresholding orchestrator.** Submits one SLURM job per VAD interval (via `spike_thresholding_worker.py`) to bandpass filter NS5 microwire channels (300–6000 Hz), compute the MAD noise scale per channel, and solve for the bisection threshold _k_ such that the global firing rate ≈ 20 Hz. Outputs a binary spike train (1 kHz) per interval. | Methods §Spike Auto-thresholding |
| `spike_thresholding_worker.py` | SLURM worker implementing the bisection autothresholding algorithm (Butterworth bandpass + 60 Hz notch, MAD noise scale, refractory period of 1 ms). | Methods §Spike Auto-thresholding |
| `assemble_word_counts_frs.ipynb` | Loads per-interval binary spike trains and the assembled transcript, aligns spikes to word intervals, computes Gaussian-smoothed firing rates (σ = 50 ms), and saves word-level spike count and firing rate matrices (`word_counts.npy`, `word_frs.npy`) used by all downstream models. | Methods §Spike Auto-thresholding; Methods §XGBoost Decoder |

---

## generate\_embeddings/

Computes GPT-2-large last-layer semantic embeddings for every word in the transcript.

| File | Description | Paper Section |
|------|-------------|---------------|
| `generate_embeddings_gpt2.ipynb` | **GPT-2 embedding orchestrator.** Submits one GPU SLURM job per eligible VAD interval to `gpt2_embedding_worker.py`. Each worker generates last-layer hidden states from GPT-2-large for all words, using a rolling context window of up to 200 preceding words. Concatenates per-interval results into `{patient}_gpt2_embeddings.npy`. | Methods §Transcript Assembly (GPT-2 embeddings) |
| `gpt2_embedding_worker.py` | GPU SLURM worker that runs GPT-2-large inference with KV-cache for efficiency, outputting the last hidden state for each word in an interval. | Methods §Transcript Assembly |

---

## standard\_encoding\_analysis/

Poisson GLM encoding models predicting unit spike counts from GPT-2 embeddings, run across multiple speech conditions and temporal granularities.

| File | Description | Paper Section |
|------|-------------|---------------|
| `word_level_duration_cv_filtered_speech.ipynb` | **Primary encoding orchestrator (whole-stay, quality-filtered, all speech).** Submits one SLURM job per patient to `poisson_glm_worker.py`. Produces pseudo-R² values per unit, used in most encoding comparisons. | Results §Encoding; **Figure 3A, 3C** |
| `word_level_duration_cv_patient_speech.ipynb` | Encoding orchestrator restricted to quality-filtered **patient-only** speech (requires `_transcripts_annotated.csv` from LR-ASD). | Results §Encoding §patient speech; **Figure 3B** |
| `word_level_duration_cv_all_n.ipynb` | Encoding orchestrator on **all unfiltered** words (no quality filtration). Used as an unfiltered baseline. | Results §Encoding (unfiltered baseline) |
| `word_level_duration_cv_filtered_speech_per_day.ipynb` | **Per-day** quality-filtered encoding: one SLURM job per (patient, date). | Results §Encoding §variability over time; **Figure 3D** |
| `word_level_duration_cv_patient_speech_per_day.ipynb` | Per-day patient-speech encoding. | Results §Encoding; **Figure 3D, 3E** |
| `word_level_duration_cv_all_n_per_day.ipynb` | Per-day unfiltered encoding baseline. | Results §Encoding (per-day baseline) |
| `word_level_duration_cv_filtered_speech_per_epoch.ipynb` | **Per-2-hour-epoch** quality-filtered encoding (blocks: 09–11, 11–13, …, 21–23 CT). Used for attention/engagement analysis. | Results §Encoding §variability over time; **Figure 3E, 4G** |
| `word_level_duration_cv_patient_speech_per_epoch.ipynb` | Per-2-hour-epoch patient-speech encoding. | Results §Encoding §variability over time; **Figure 3E** |
| `poisson_glm_worker.py` | SLURM worker implementing the Poisson GLM: PCA reduction (1280→100 PCs), L2-regularized negative log-likelihood optimization in PyTorch (L-BFGS), 5-fold outer CV with inner CV for regularization hyperparameter selection, and permutation testing (50 shuffles). | Methods §Poisson GLM; Methods §Pseudo R²; Methods §Permutation Testing |
| `plot_encoding_results.py` | Utility functions for plotting encoding results (used by figure generation notebooks). | — |
| `plot_encoding_results_comparison.py` | Utility functions for plotting encoding condition comparisons. | — |

---

## standard\_decoding\_analysis/

XGBoost semantic category classifiers run across multiple speech conditions, temporal granularities, and brain regions.

| File | Description | Paper Section |
|------|-------------|---------------|
| `scat_classifier_sampled_nocv_filtered_speech.ipynb` | **Primary decoding orchestrator (whole-stay, quality-filtered, all speech).** Submits one SLURM job per patient/resample. Resamples are stratified across semantic categories (20 total), with hyperparameter search on the first 3. | Results §Decoding; **Figure 4A, 4C** |
| `scat_classifier_sampled_nocv_filtered_patient_speech.ipynb` | Decoding restricted to quality-filtered **patient-only** speech. | Results §Decoding §patient speech; **Figure 4B** |
| `scat_classifier_sampled_nocv.ipynb` | Decoding on unfiltered all-speech (no quality filtration). Baseline variant. | Results §Decoding (baseline) |
| `scat_classifier_sampled_nocv_filtered_speech_per_day.ipynb` | **Per-day** quality-filtered decoding. | Results §Decoding §variability over time; **Figure 4D** |
| `scat_classifier_sampled_nocv_patient_speech_per_day.ipynb` | Per-day patient-speech decoding. | Results §Decoding; **Figure 4D** |
| `scat_classifier_sampled_nocv_filtered_speech_per_epoch.ipynb` | **Per-2-hour-epoch** quality-filtered decoding. | Results §Decoding §variability over time; **Figure 4F, 4G** |
| `scat_classifier_sampled_nocv_patient_speech_per_epoch.ipynb` | Per-2-hour-epoch patient-speech decoding. | Results §Decoding; **Figure 4F** |
| `scat_classifier_region_per_day.ipynb` | Per-day decoding split by **brain region** (HPC, AMY, OFC, THAL, ACC, PCC). Subsamples 8 units per region across 5 brain resamples × 10 semantic resamples. | Results §Decoding §brain regions; **Figure 4H** |
| `prepare_semantic_cluster_predictions.ipynb` | Builds per-patient fastText word embeddings and trains an XGBoost classifier to assign semantic category labels to all words. Produces `fasttext_word_embeddings.npy` and `semantic_cluster_predictions.npy`. | Methods §Semantic Category Classifier |
| `scat_classifier_worker.py` | SLURM worker implementing the XGBoost semantic category decoder: Gaussian-smoothed population firing rates as features, randomized hyperparameter search, class-balanced downsampling, and multi-resample averaging. | Methods §XGBoost Decoder |
| `scat_classifier_region_worker.py` | SLURM worker for per-region decoding; handles unit subsampling per brain region. | Methods §XGBoost Decoder; Results §Decoding §brain regions |
| `semantic_cluster_worker.py` | SLURM worker that computes fastText embeddings and semantic cluster predictions for a single patient. | Methods §Semantic Category Classifier |
| `plot_decoding_results.py` | Utility functions for plotting decoding results (used by figure generation notebooks). | — |

---

## convo\_behav\_comparison/

Ground-truth evaluation of transcription accuracy and diarization quality against manually annotated conversation-task recordings. Produces the accuracy benchmarks in Figure 2.

| File | Description | Paper Section |
|------|-------------|---------------|
| `convo_accuracy_evaluation.ipynb` | **Transcription accuracy evaluation.** Matches each manually-annotated Praat sentence to the best-matching WhisperX sentence via fuzzy token matching + Levenshtein WER. Reports match rate, WER similarity, SBERT cosine similarity, and sentence/word timing errors. | Results §Dataset §Ground-truth accuracy; **Figure 2D–F** |
| `convo_diarization_evaluation.ipynb` | **Diarization accuracy evaluation.** Compares LR-ASD patient-speaking predictions against Praat-diarized speaker labels. Reports precision, recall, F1, and compares against base-rate performance per patient. | Results §Dataset §diarization; **Figure 2G** |
| `align_ref_timing.ipynb` | Corrects timing misalignment between Praat reference transcripts and WhisperX hypothesis timestamps for specific patients. Estimates a linear clock-correction mapping from matched sentence pairs (WER-only, no timing constraint) and applies it before evaluation. | Methods §Sentence Matching Algorithm (preprocessing) |
| `run_convo_asd.ipynb` | Submits one SLURM job per camera for the conversation-task recordings of `anilu_comparison` patients, running the full ASD/video-sync pipeline (`video-sync-nbu`) to produce LR-ASD speaker scores. | Methods §Active Speaker Detection (ground-truth task) |
| `convo_asd_worker.py` | SLURM worker for running ASD on a single conversation-task camera. | Methods §Active Speaker Detection |
| `sbert_worker.py` | Computes SBERT (`all-mpnet-base-v2`) sentence embeddings for matched sentence pairs in `convo_accuracy_evaluation.ipynb`, used to calculate semantic cosine similarity. | Results §Dataset §Quantifying semantic precision |

---

## decimation\_analysis/

Measures how encoding model performance scales with the size of the training dataset (Figure 5).

| File | Description | Paper Section |
|------|-------------|---------------|
| `decimation_orchestrator_per_day.ipynb` | **Primary decimation orchestrator (per-day).** Submits one SLURM job per (patient, date, portion, sample) combination. Portions range from 1% to 100%; 20–100 random subsamples per portion per day. Fits the Hill equation to performance-vs-portion curves. | Results §Decimation; **Figure 5A–C** |
| `decimation_orchestrator.ipynb` | Whole-stay decimation (same analysis as above but on the full recording rather than individual days). Used for supplementary decimation panels. | Results §Decimation (supplementary) |
| `decimation_glm_worker.py` | SLURM worker that runs the Poisson GLM on a randomly subsampled word set for a single (patient, date, portion, sample) combination. | Methods §Poisson GLM; Results §Decimation |

---

## functional\_drift/

Evaluates how well encoding and decoding models trained on one day generalize to future recording days (Figure 6).

| File | Description | Paper Section |
|------|-------------|---------------|
| `encoding_drift.ipynb` | **Encoding drift orchestrator.** Phase 1: one SLURM training job per (patient, train_date); Phase 2: one test job per (patient, train_date, test_date). Tests day-to-day generalization of Poisson GLM encoding models with permutation testing. | Results §Functional Drift; **Figure 6A, 6B, 6E** |
| `decoding_drift.ipynb` | **Decoding drift orchestrator.** Same cross-day generalization test structure for XGBoost semantic category decoders. | Results §Functional Drift; **Figure 6C, 6D, 6E** |
| `encoding_singlepca_drift.ipynb` | Encoding drift orchestrator using a **shared global PCA basis** (fit once on all training data). Enables direct cosine-distance comparison of per-day tuning function weight vectors. | Results §Functional Drift; **Figure 6F** |
| `encoding_singlepca_epoch_drift.ipynb` | Same as `encoding_singlepca_drift.ipynb` but at 2-hour epoch granularity, for finer-grained tuning-function drift analysis. | Results §Functional Drift; **Figure 6F** |
| `encoding_drift_train_worker.py` | SLURM worker: fits a full-day Poisson GLM encoding model for one (patient, train_date) pair and saves weights + alpha. | Methods §Poisson GLM; Results §Functional Drift |
| `encoding_drift_test_worker.py` | SLURM worker: evaluates saved encoding model on a held-out test date. Reports pseudo-R² and permutation p-values. | Results §Functional Drift |
| `decoding_drift_train_worker.py` | SLURM worker: fits a full-day XGBoost decoder for one (patient, train_date, resample) combination. | Methods §XGBoost Decoder; Results §Functional Drift |
| `decoding_drift_test_worker.py` | SLURM worker: evaluates saved decoder on a held-out test date. Reports accuracy and permutation p-values. | Results §Functional Drift |
| `encoding_singlepca_train_worker.py` | SLURM worker for per-day encoding model training under a shared global PCA projection. | Results §Functional Drift |
| `encoding_singlepca_test_worker.py` | SLURM worker for cross-day evaluation under the shared PCA basis. | Results §Functional Drift |
| `encoding_singlepca_global_bundle_worker.py` | SLURM worker that fits the global PCA on all words across all training days for a patient. | Results §Functional Drift |
| `encoding_singlepca_epoch_train_worker.py` | SLURM worker for per-epoch encoding training under shared PCA. | Results §Functional Drift |
| `encoding_singlepca_epoch_bundle_worker.py` | SLURM worker that bundles per-epoch GLM results for drift computation. | Results §Functional Drift |

---

## controlled\_comparison/

Encoding and decoding analyses on the two ground-truth controlled task datasets (**conversation** and **podcast**), including automated variants for benchmarking. Results appear in Figure 3C and Figure 4C.

### convo\_encoding\_comparison/

Poisson GLM encoding models trained on the manually annotated conversation task.

| File | Description | Paper Section |
|------|-------------|---------------|
| `word_level_duration_cv_all_n.ipynb` | Encoding on conversation task with **manually sorted spikes** and **hand-labelled transcripts**. | Results §Encoding §Comparison with controlled tasks; **Figure 3C** |
| `word_level_duration_cv_all_n_auto.ipynb` | Encoding on conversation task with **autothresholded spikes** (automated neural processing). | Results §Encoding §Comparison with controlled tasks; **Figure 3C** |
| `word_level_duration_cv_all_n_whisperx.ipynb` | Encoding on conversation task with autothresholded spikes + **WhisperX transcripts** (fully automated). | Results §Encoding §Comparison with controlled tasks; **Figure 3C** |
| `assemble_auto_behaviral_data.ipynb` | Assembles autothresholded spike counts aligned to word timestamps for the conversation task. | Methods §Spike Auto-thresholding (controlled task) |
| `get_auto_word_spikes.ipynb` | Extracts per-word autothresholded spike counts for the conversation task comparison. | Methods §Spike Auto-thresholding (controlled task) |
| `format_whisper_outs.ipynb` | Formats raw WhisperX JSON outputs for the conversation task into the word-level DataFrame format expected by encoding models. | Methods §Transcript Assembly (controlled task) |
| `get_audio_waveforms.ipynb` | Extracts audio waveforms from the conversation task recordings for quality inspection. | — |
| `spike_thresholding.ipynb` | Runs spike auto-thresholding on conversation task NS5 recordings. | Methods §Spike Auto-thresholding (controlled task) |
| `word_level_duration_cv.ipynb` | Earlier encoding prototype for the conversation task. Possibly stale — see note below. | — |

### convo\_scat\_comparison/

XGBoost semantic category decoders on the conversation task.

| File | Description | Paper Section |
|------|-------------|---------------|
| `scat_classifier_sampled_nocv.ipynb` | Decoding on conversation task with **manually sorted spikes** and **hand-labelled transcripts**. | Results §Decoding §Comparison with ground-truth datasets; **Figure 4C** |
| `scat_classifier_sampled_nocv_auto.ipynb` | Decoding with **autothresholded spikes**. | Results §Decoding §Comparison with ground-truth datasets; **Figure 4C** |
| `scat_classifier_sampled_nocv_whisperx.ipynb` | Decoding with autothresholded spikes + **WhisperX transcripts**. | Results §Decoding §Comparison with ground-truth datasets; **Figure 4C** |
| `assemble_model_inputs.ipynb` | Assembles model inputs (firing rates + semantic cluster labels) for the conversation decoding comparison. | Methods §XGBoost Decoder (controlled task) |
| `scat_xgboost.ipynb` | Early decoding experiment (no class balancing). Possibly stale — see note below. | — |
| `scat_xgboost_balanced.ipynb` | Early decoding experiment with class balancing. Possibly stale — see note below. | — |
| `scat_classifier_sampled_nocv_stratified_speakerid.ipynb` | Decoding experiment stratifying by speaker ID. Not in main results. Possibly stale. | — |
| `scat_classifier_sampled_nocv_with_speakerid.ipynb` | Decoding with speaker ID as an additional feature. Not in main results. Possibly stale. | — |

### podcast\_encoding\_comparison/

Poisson GLM encoding models trained on the podcast listening task. Mirrors `convo_encoding_comparison/` in structure.

| File | Description | Paper Section |
|------|-------------|---------------|
| `word_level_duration_cv_all_n.ipynb` | Encoding on podcast task with **manually sorted spikes** and hand-labelled transcripts. | Results §Encoding §Comparison with controlled tasks; **Figure 3C** |
| `word_level_duration_cv_all_n_auto.ipynb` | Encoding on podcast task with **autothresholded spikes**. | Results §Encoding §Comparison with controlled tasks; **Figure 3C** |
| `word_level_duration_cv_all_n_whisperx.ipynb` | Encoding on podcast task fully automated (autothresholded spikes + WhisperX). | Results §Encoding §Comparison with controlled tasks; **Figure 3C** |
| `assemble_auto_behaviral_data.ipynb` | Assembles autothresholded spike counts for the podcast task. | Methods §Spike Auto-thresholding (controlled task) |
| `get_auto_word_spikes.ipynb` | Extracts autothresholded spike counts per word for podcast task. | Methods §Spike Auto-thresholding (controlled task) |
| `format_whisper_outs.ipynb` | Formats WhisperX outputs for the podcast task. | Methods §Transcript Assembly (controlled task) |
| `get_audio_waveforms.ipynb` | Extracts audio waveforms from podcast recordings. | — |
| `spike_thresholding.ipynb` | Spike auto-thresholding on podcast task NS5 recordings. | Methods §Spike Auto-thresholding (controlled task) |
| `word_level_duration_cv.ipynb` | Earlier encoding prototype for the podcast task. Possibly stale — see note below. | — |

### podcast\_scat\_comparison/

XGBoost semantic category decoders on the podcast task.

| File | Description | Paper Section |
|------|-------------|---------------|
| `scat_classifier_sampled_nocv.ipynb` | Decoding on podcast task with **manually sorted spikes**. | Results §Decoding §Comparison with ground-truth datasets; **Figure 4C** |
| `scat_classifier_sampled_nocv_whisperx.ipynb` | Decoding on podcast task with WhisperX transcripts (automated). | Results §Decoding §Comparison with ground-truth datasets; **Figure 4C** |
| `assemble_model_inputs.ipynb` | Assembles model inputs for podcast decoding. | Methods §XGBoost Decoder (controlled task) |
| `scat_xgboost.ipynb` | Early decoding experiment (no balancing). Possibly stale. | — |
| `scat_xgboost_balanced.ipynb` | Early decoding experiment with class balancing. Possibly stale. | — |

---

## figure\_generation/

Generates all main figures. Each notebook loads pre-computed model results (`.pkl` files) and produces the final SVG/PDF panels.

| File | Description | Paper Section |
|------|-------------|---------------|
| `figure_2.ipynb` | **Figure 2.** Panels: (A) words/hours per patient bar plots, (B) UMAP of word semantic space, (C) daily speech cycle, (D) transcription match rate, (E) example sentence pairs, (F) error distributions (WER, SBERT, timing), (G) diarization accuracy. | Results §Dataset; **Figure 2** |
| `figure_3.ipynb` | **Figure 3.** Panels: (A) per-patient encoding performance, (B) all vs. patient speech, (C) naturalistic vs. controlled comparisons, (D) whole-stay vs. per-day, (E) epoch performance vs. speech fraction, (F) performance by brain region. | Results §Encoding; **Figure 3** |
| `figure_4.ipynb` | **Figure 4.** Panels: (A) per-patient decoding accuracy, (B) all vs. patient speech, (C) naturalistic vs. controlled, (D) whole-stay vs. per-day, (E) accuracy vs. unit population size, (F) epoch decoding vs. speech fraction, (G) epoch stability comparison, (H) decoding by brain region. | Results §Decoding; **Figure 4** |
| `figure_5.ipynb` | **Figure 5.** Panels: (A) pseudo-R² vs. dataset portion (Hill curve), (B) pseudo-R² vs. word count (2D histogram + Hill curve), (C) fraction significant units vs. dataset portion. | Results §Decimation; **Figure 5** |
| `figure_6.ipynb` | **Figure 6.** Panels: (A) encoding drift across days, (B) fraction significant encoding units vs. days, (C) decoding drift across days, (D) fraction significant decoding models vs. days, (E) slope distributions (encoding vs. decoding), (F) tuning function cosine distance vs. time. | Results §Functional Drift; **Figure 6** |
| `add_region_panels.py` | Post-processing script that adds brain region overlay panels to assembled figure files. | — |
| `fix_fig3_ylim.py` | Post-processing script: adjusts y-axis limits on Figure 3 panels. | — |
| `fix_panel_e_f.py` | Post-processing script: fixes panel E and F layout/formatting in an assembled figure. | — |
| `fix_panels_and_ylim.py` | Post-processing script: combined panel layout and y-limit corrections. | — |
| `fix_wls_and_region.py` | Post-processing script: fixes WLS fit lines and region panel formatting. | — |
| `rebuild_fig4.py` | Post-processing script: rebuilds Figure 4 from saved panel components with corrected formatting. | — |

---

## Potentially Stale Notebooks

The following notebooks are flagged as potentially stale or no longer part of the primary analysis pipeline. Please verify before use:

| Notebook | Reason |
|----------|--------|
| `speech_extraction/identify_channel_discreps.ipynb` | Empty — no code cells |
| `speech_extraction/identify_clock_resets.ipynb` | Empty — no code cells |
| `video_processing/sync_video_patient.ipynb` | Superseded by `run_video_sync_patient.ipynb` (uses legacy `batch_sync_patient` API) |
| `video_processing/add_nsp1_data.ipynb` | Targets legacy `vad_out` directory layout |
| `video_processing/add_nsp1_data_all_patients.ipynb` | Targets legacy `vad_out` directory layout |
| `controlled_comparison/convo_encoding_comparison/word_level_duration_cv.ipynb` | Early encoding prototype; no clear differentiation from `_all_n` version |
| `controlled_comparison/podcast_encoding_comparison/word_level_duration_cv.ipynb` | Early encoding prototype; same as above |
| `controlled_comparison/convo_scat_comparison/scat_xgboost.ipynb` | Early decoding experiment without resampling/balancing |
| `controlled_comparison/convo_scat_comparison/scat_xgboost_balanced.ipynb` | Intermediate decoding experiment |
| `controlled_comparison/convo_scat_comparison/scat_classifier_sampled_nocv_stratified_speakerid.ipynb` | Speaker-ID experiment; not in main results |
| `controlled_comparison/convo_scat_comparison/scat_classifier_sampled_nocv_with_speakerid.ipynb` | Speaker-ID feature experiment; not in main results |
| `controlled_comparison/podcast_scat_comparison/scat_xgboost.ipynb` | Early decoding experiment |
| `controlled_comparison/podcast_scat_comparison/scat_xgboost_balanced.ipynb` | Intermediate decoding experiment |
| `create_whisper_jobs.sh` / `execute_whisper_jobs.sh` | Legacy shell-based transcription; superseded by SLURM notebook orchestration |
