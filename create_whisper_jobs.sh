#!/bin/bash
# first make sure correct environment is active
#module load Miniforge3
#conda activate speech_247_env
# get appropriate variables
patient="YFU"
base_dir="/mnt/labworlds/Hayden/Hayden_Lab/speech_247/vad_out/${patient}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before running: export HF_TOKEN=<your_token>}"
# login to huggingface cli
export HF_HOME='/scratch/tahaismail424/hf'
# huggingface-cli login --token $HF_TOKEN --add-to-git-credential
# 1. Collect all eligible audio files (excluding already-processed dirs)
find "$base_dir" -type f -name "*.wav" | while read -r audio; do
    recording_dir=$(dirname "$(dirname "$audio")")  # /.../recording*/audio/ → /.../recording*/
    transcription_dir="${recording_dir}/transcription"
    
    # Skip if already transcribed
    if [ -d "$transcription_dir" ]; then
        continue
    fi

    echo "$audio"
done > audio_jobs.txt