#!/usr/bin/env python3
"""
extract_target_embedding.py

Extracts a target speaker embedding from a given voice recording.
Usage:
    python extract_target_embedding.py --input target_voice.wav --output target_embedding.pt
"""

import argparse
import torch
import torchaudio
import numpy as np

# Import the SpeakerEncoder from so-vits-svc-fork.
# Adjust the import if needed.
from so_vits_svc_fork.modules.encoders import SpeakerEncoder

def waveform_to_mel(waveform, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256, win_length=1024):
    """
    Convert a waveform to a log-mel spectrogram.
    
    Parameters:
      waveform (Tensor): shape [channels, time]
      sample_rate (int): sample rate of the audio
      n_mels (int): number of mel bins (must match what the encoder expects)
      n_fft (int): FFT window size
      hop_length (int): hop length
      win_length (int): window length
    
    Returns:
      mel (Tensor): shape [batch, time, n_mels]
    """
    # If stereo, average the channels.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Create a MelSpectrogram transform.
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels
    )
    # Compute mel spectrogram. Expected shape: [channels, n_mels, time]
    mel = mel_transform(waveform)
    # Convert to log scale.
    mel = torch.log(mel + 1e-6)
    # Transpose to get shape: [channels, time, n_mels]
    mel = mel.transpose(1, 2)
    # NOTE: Do NOT add an extra batch dimension if the transform already returns [1, T, n_mels]
    # If waveform was mono, mel shape is [1, T, 80], which is what we want.
    return mel

def extract_embedding(input_audio_path, output_path, sample_rate=16000):
    # Load the target voice audio.
    waveform, sr = torchaudio.load(input_audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    # Convert waveform to mel spectrogram.
    mel = waveform_to_mel(waveform, sample_rate=sample_rate)
    # At this point, mel should have shape [1, T, 80].
    
    # Instantiate the speaker encoder.
    encoder = SpeakerEncoder()  # Adjust this if your encoder needs a pretrained checkpoint.
    encoder.eval()
    encoder = encoder.to("cpu")
    
    with torch.no_grad():
        # Compute the speaker embedding using the mel spectrogram.
        embedding = encoder(mel)
    
    torch.save(embedding, output_path)
    print(f"Target speaker embedding saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract target speaker embedding")
    parser.add_argument("--input", type=str, required=True, help="Path to target voice WAV file")
    parser.add_argument("--output", type=str, default="target_embedding.pt", help="Output embedding file")
    args = parser.parse_args()
    
    extract_embedding(args.input, args.output)