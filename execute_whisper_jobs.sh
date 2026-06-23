#!/bin/bash
# first make sure correct environment is active
export HF_HOME='/scratch/tahaismail424/hf'

HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before running: export HF_TOKEN=<your_token>}"
export HF_HOME='/scratch/tahaismail424/hf'
hf auth login --token $HF_TOKEN --add-to-git-credential

# 2. Define the processing function
run_whisperx() {
    audio="$1"
    recording_dir=$(dirname "$(dirname "$audio")")
    transcription_dir="${recording_dir}/transcription"
    mkdir -p "$transcription_dir"

    basename=$(basename "$audio" .wav)
    out_dir="${transcription_dir}/${basename}"
    mkdir -p "$out_dir"

    whisperx "$audio" \
        --model large-v2 \
        --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \
        --diarize \
        --highlight_words True \
        --output_dir "$out_dir" \
        --language English \
        &> "${out_dir}/log.txt"
}
export -f run_whisperx


# 3. Run in parallel
parallel --jobs 6 run_whisperx :::: audio_jobs.txt