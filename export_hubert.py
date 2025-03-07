#!/usr/bin/env python3

import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

def export_hubert(
    model_name="./hubert-base-ls960",  # Local path to the downloaded model folder
    output_file="hubert_scripted_traced.pt",
    sample_rate=16000
):
    """
    Exports a Hugging Face 'facebook/hubert-base-ls960' model (downloaded locally)
    to a TorchScript file.
    
    :param model_name: Local folder path for the model (with config.json and pytorch_model.bin).
    :param output_file: Output filename for the TorchScript model.
    :param sample_rate: Sample rate for creating dummy input.
    """
    
    print(f"Loading model and feature extractor from '{model_name}'...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    
    # Disable gradient checkpointing if applicable
    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = False

    print("Creating dummy input for tracing...")
    # Create a dummy waveform of 1 second (sample_rate samples)
    dummy_waveform = torch.randn(sample_rate, dtype=torch.float32)
    inputs = feature_extractor(dummy_waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs["input_values"]  # shape: [1, seq_length]

    print("Tracing the model with TorchScript (using strict=False)...")
    with torch.no_grad():
        # Use strict=False to allow dictionary outputs from the model
        traced_model = torch.jit.trace(model, (input_values,), strict=False)

    print(f"Saving TorchScript model to '{output_file}'...")
    traced_model.save(output_file)
    print("Export completed successfully!")

if __name__ == "__main__":
    export_hubert()