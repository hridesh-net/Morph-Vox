"""
offline_inference.py

Offline voice conversion using a custom target embedding with so-vits-svc-fork.
This script:
  - Loads the conversion model (SynthesizerTrn), HuBERT encoder, and NSF-HIFIGAN vocoder.
  - Loads a custom target embedding (target_embedding.pt) and forces the model to use it.
  - Reads an input WAV file, converts the input voice to the target voice, and saves the output.
  
Usage:
  python offline_inference.py --input input.wav --output output.wav --transpose 0 --target_embedding target_embedding.pt
"""

import argparse
import json
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
import torch.nn as nn

# Import the Svc class from so_vits_svc_fork inference.
from so_vits_svc_fork.inference.core import Svc


# Define a custom module that always returns the constant target embedding.
class ConstantEmbedding(nn.Module):
    def __init__(self, target_embedding: torch.Tensor):
        super().__init__()
        # target_embedding should have shape (1, gin_channels)
        self.target_embedding = target_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is ignored; instead, we return the target embedding repeated for the batch.
        batch_size = x.shape[0]
        # Return a tensor of shape (batch_size, 1, gin_channels)
        return self.target_embedding.unsqueeze(1).repeat(batch_size, 1, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Offline voice conversion using a custom target embedding."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output WAV file")
    parser.add_argument("--transpose", type=int, default=0, help="Transpose value (in semitones)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/44k/config.json",
        help="Path to the conversion model config.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/44k/G_0.pth",
        help="Path to the conversion model checkpoint (G_0.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g., cpu or cuda)",
    )
    parser.add_argument(
        "--target_embedding",
        type=str,
        default="t_emb.pt",
        help="Path to the custom target speaker embedding file",
    )
    args = parser.parse_args()

    # Initialize the Svc model.
    svc_model = Svc(net_g_path=args.checkpoint, config_path=args.config, device=args.device)
    print("Model loaded. Target sample rate:", svc_model.target_sample)

    # Load custom target embedding.
    target_emb = None
    target_emb_path = Path(args.target_embedding)
    if target_emb_path.exists():
        target_emb = torch.load(str(target_emb_path), map_location=args.device)
        # Ensure the target embedding has shape (1, gin_channels)
        if target_emb.ndim == 1:
            target_emb = target_emb.unsqueeze(0)
        print("Custom target embedding loaded from", args.target_embedding)
    else:
        print("No target embedding found at", args.target_embedding)
    
    # If a custom target embedding is provided, override the model's speaker embedding.
    if target_emb is not None:
        svc_model.net_g.emb_g = ConstantEmbedding(target_emb)
        print("Model's speaker embedding lookup overridden with custom target embedding.")

    # Load input audio.
    waveform, sr = torchaudio.load(args.input)
    # If multi-channel, use only the first channel.
    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)
    print(f"Loaded {args.input} (SR={sr}).")
    
    # Convert waveform to a NumPy array in float32.
    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    # Run inference.
    # When using a custom target embedding, the speaker parameter is ignored.
    converted_audio, length = svc_model.infer(
        speaker="0",  # This value is ignored due to our override.
        transpose=args.transpose,
        audio=audio_np,
        cluster_infer_ratio=0,
        auto_predict_f0=False,
        noise_scale=0.4,
        f0_method="dio",
    )
    # Ensure converted_audio is a NumPy array.
    converted_audio = converted_audio.cpu().numpy()

    # Save the converted audio.
    sf.write(args.output, converted_audio, svc_model.target_sample, subtype="PCM_16")
    print(f"Conversion complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()