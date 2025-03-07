#!/usr/bin/env python3
"""
extract_target_embedding_resemblyzer.py

Extracts a robust target speaker embedding from a given target voice WAV file using Resemblyzer.
This script:
  - Loads the target audio using torchaudio.
  - Averages multi-channel audio to mono.
  - Resamples the audio to the desired sample rate (default 16000 Hz) if needed.
  - Preprocesses the waveform using Resemblyzer's preprocess_wav.
  - Computes the speaker embedding with the pretrained VoiceEncoder.
  - Saves the embedding as a PyTorch tensor.

Usage:
  python extract_target_embedding_resemblyzer.py --input target_voice.wav --output target_embedding.pt
"""

import argparse
import torch
import torchaudio
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav

def extract_embedding(audio_path: str, output_path: str, sample_rate: int = 16000) -> None:
    # Load the audio using torchaudio.
    wav, sr = torchaudio.load(audio_path)
    # If multi-channel, average to mono.
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Convert tensor to numpy array.
    wav = wav.squeeze(0).numpy()
    
    # Resample if the sample rate is not the desired one.
    print(sr)
    # if sr != sample_rate:
    #     wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    
    # Preprocess the waveform as expected by Resemblyzer.
    # Note: We no longer pass sampling_rate as a keyword argument.
    wav = preprocess_wav(wav)
    
    # Instantiate the VoiceEncoder (pretrained model).
    encoder = VoiceEncoder()
    # Compute the speaker embedding.
    embedding = encoder.embed_utterance(wav)
    
    # Convert the embedding to a PyTorch tensor.
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
    
    # Save the embedding.
    torch.save(embedding_tensor, output_path)
    print(f"Target speaker embedding saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract a target speaker embedding using Resemblyzer."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to target voice WAV file")
    parser.add_argument("--output", type=str, default="target_embedding.pt", help="Output embedding file")
    args = parser.parse_args()
    
    extract_embedding(args.input, args.output)

if __name__ == "__main__":
    main()