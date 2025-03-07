#!/usr/bin/env python3
"""
realtime_inference_buffered_nr.py

Real-time voice conversion using so-vits-svc-fork and PyAudio with buffering and noise reduction.
This script:
  - Loads the conversion model (Svc) along with an optional custom target embedding.
  - Captures audio from the microphone and accumulates it in a buffer until a minimum length is reached.
  - Applies noise reduction to each chunk before buffering.
  - Processes the buffered audio through the conversion model.
  - Plays the converted audio in real time.

Usage:
  python realtime_inference_buffered_nr.py --checkpoint logs/44k/G_0.pth --config configs/44k/config.json --device cpu --sample_rate 44100 --target_embedding target_embedding.pt --transpose 0
"""

import argparse
import torch
import numpy as np
import pyaudio
import torch.nn as nn
from pathlib import Path
from so_vits_svc_fork.inference.core import Svc
import noisereduce as nr

# Define a custom module that always returns the constant target embedding.
class ConstantEmbedding(nn.Module):
    def __init__(self, target_embedding: torch.Tensor):
        super().__init__()
        self.target_embedding = target_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.target_embedding.unsqueeze(1).repeat(batch_size, 1, 1)

def main():
    parser = argparse.ArgumentParser(description="Real-time voice conversion with buffering and noise reduction.")
    parser.add_argument("--checkpoint", type=str, default="logs/44k/G_0.pth", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/44k/config.json", help="Path to model config file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--target_embedding", type=str, default="", help="Path to custom target embedding (optional)")
    parser.add_argument("--transpose", type=int, default=0, help="Transpose value in semitones")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Audio I/O sample rate")
    parser.add_argument("--chunk", type=int, default=1024, help="Chunk size (samples) for reading from mic")
    parser.add_argument("--min_buffer", type=int, default=8192, help="Minimum buffer size (samples) for inference")
    args = parser.parse_args()

    # Load the Svc model.
    svc_model = Svc(net_g_path=args.checkpoint, config_path=args.config, device=args.device)
    print("Model loaded. Target sample rate:", svc_model.target_sample)

    # Optional: Override speaker embedding with custom target embedding.
    if args.target_embedding:
        emb_path = Path(args.target_embedding)
        if emb_path.exists():
            target_emb = torch.load(str(emb_path), map_location=args.device)
            if target_emb.ndim == 1:
                target_emb = target_emb.unsqueeze(0)
            svc_model.net_g.emb_g = ConstantEmbedding(target_emb)
            print("Custom target embedding override activated.")
        else:
            print("Warning: target embedding file not found. Using internal speaker embedding.")
    else:
        print("No custom embedding provided, using internal speaker embedding.")

    # Setup PyAudio for input (microphone) and output (speakers).
    p = pyaudio.PyAudio()
    CHUNK = args.chunk
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = args.sample_rate

    stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

    # Buffer to accumulate audio for inference.
    buffer = np.array([], dtype=np.float32)
    MIN_BUFFER = args.min_buffer

    print("Starting real-time conversion with buffering and noise reduction... Press Ctrl+C to stop.")
    try:
        while True:
            # Read a chunk from the microphone.
            audio_chunk = stream_in.read(CHUNK, exception_on_overflow=False)
            # Convert to float32 numpy array and normalize to [-1, 1].
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # Apply noise reduction to the chunk.
            audio_np = nr.reduce_noise(y=audio_np, sr=RATE)

            # Accumulate the processed chunk in the buffer.
            buffer = np.concatenate((buffer, audio_np))

            # If the buffer length exceeds the minimum required for inference, process it.
            if len(buffer) >= MIN_BUFFER:
                converted_tensor, _ = svc_model.infer(
                    speaker="0",  # This value is ignored if using ConstantEmbedding.
                    transpose=args.transpose,
                    audio=buffer,
                    cluster_infer_ratio=0,
                    auto_predict_f0=False,
                    noise_scale=0.4,
                    f0_method="dio",
                )
                converted_np = converted_tensor.cpu().numpy()
                converted_np = np.clip(converted_np, -1.0, 1.0)
                converted_int16 = (converted_np * 32767.0).astype(np.int16)
                stream_out.write(converted_int16.tobytes())
                # Clear the buffer (you might later implement overlap or crossfade for smoother output).
                buffer = np.array([], dtype=np.float32)
    except KeyboardInterrupt:
        print("Stopping real-time conversion.")
    finally:
        stream_in.stop_stream()
        stream_out.stop_stream()
        stream_in.close()
        stream_out.close()
        p.terminate()

if __name__ == "__main__":
    main()