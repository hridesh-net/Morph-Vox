# Real-Time Voice Conversion System

This project implements a CPU‑based, real‑time voice conversion system using [so‑vits‑svc‑fork](https://github.com/voicepaw/so-vits-svc-fork). The system captures live microphone audio, converts it to a target voice using a conversion model, and outputs the converted audio in real time. It also provides offline inference and noise reduction features, and supports using a custom target embedding.

## Features

- **Real-Time Conversion:**  
  Capture live audio from your microphone and convert it on the fly.
- **Offline Inference:**  
  Test conversion quality using recorded WAV files.
- **Custom Target Embedding:**  
  Optionally override the model’s learned speaker embedding with a custom target embedding extracted from a clean target voice.
- **Noise Reduction:**  
  Applies noise reduction (using noisereduce) to improve speech clarity.
- **Buffering for Stability:**  
  Accumulates audio chunks into a buffer to ensure inputs are long enough for the model.
- **Training Pipeline Support:**  
  (Optional) Follow the provided training pipeline to create your own model checkpoint.

## Directory Structure
```bash
real_time_voice_conversion/
├── configs/
│   └── 44k/
│       └── config.json         # Model configuration (for training/inference)
├── logs/
│   └── 44k/
│       └── G_0.pth             # Model checkpoint (pre-trained or custom trained)
├── models/
│   └── nsf_hifigan/
│       ├── config.json         # NSF-HIFIGAN vocoder configuration
│       └── model/
│           └── model.pt        # NSF-HIFIGAN vocoder weights
├── dataset_raw/                # Folder for raw training data (if training)
├── target_embedding.pt         # Custom target embedding file (optional)
├── extract_target_embedding_resemblyzer.py   # Script to extract target embeddings using Resemblyzer
├── offline_inference.py        # Offline voice conversion testing script
├── realtime_inference_buffered_nr.py  # Real-time conversion script with buffering and noise reduction
├── README.md                   # This file
└── requirements.txt            # Required Python packages
```
## Requirements

- Python 3.10 or later
- [so‑vits‑svc‑fork](https://pypi.org/project/so-vits-svc-fork/)
- PyTorch and torchaudio
- PyAudio (requires PortAudio on macOS/Linux)
- noisereduce and librosa
- Resemblyzer (for target embedding extraction, optional)

### Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Installation
1. Clone the Repository:

```bash
git clone <your_repo_url>
cd real_time_voice_conversion
```

2. Set Up a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

**Note**: On macOS, you might need to install PortAudio:
```bash
brew install portaudio
```


## Usage

1. ### Extracting a Target Embedding (Optional)

For robust target embeddings, you can use Resemblyzer. Run:

python extract_target_embedding_resemblyzer.py --input path/to/target_voice.wav --output target_embedding.pt

Make sure the target voice recording is clean and high-quality for best results.

2. ### Offline Inference

Test your conversion pipeline on a recorded WAV file:

python offline_inference.py --input path/to/input.wav --output path/to/output.wav --transpose 0 --target_embedding target_embedding.pt

This script loads the model, optionally overrides the speaker embedding with your custom target embedding, converts the input voice, and saves the result.

3. ### Real-Time Inference

For real-time conversion (e.g., for use in Zoom or for voice-overs), run:

```bash
python realtime_inference_buffered_nr.py \
    --checkpoint logs/44k/G_0.pth \
    --config configs/44k/config.json \
    --device cpu \
    --sample_rate 44100 \
    --target_embedding target_embedding.pt \
    --transpose 0
```

**This script:**
- Captures audio from your microphone.
- Applies noise reduction to each chunk.
- Buffers audio until a minimum length is reached.
- Runs inference through the conversion model.
- Outputs the converted audio in real time.

> To use the converted audio in calls, consider routing the output to a virtual audio device (e.g., using Loopback or BlackHole on macOS).

## Training Your Own Model

If you wish to train your own voice conversion model:
1.	Prepare Your Dataset:
Organize your recordings in dataset_raw/ with a subfolder for each speaker (e.g., dataset_raw/target_speaker/).
2.	Preprocess the Data:
Run the provided commands in order:
    - Pre-resample:

    ```bash
    svc pre-resample
    ```

	- Generate Configuration:

    ```bash
    svc pre-config
    ```

	- Extract Content Features:

    ```bash
    svc pre-hubert
    ```

3.	Train the Model:
    - Start training with:

    ```bash
    svc train -t
    ```

> To set a specific number of epochs (e.g., 50), edit the "epochs" value in configs/44k/config.json accordingly.

## Troubleshooting
- TensorBoard:
During training, TensorBoard may open in your browser. If no dashboards appear immediately, wait a few steps and refresh.
- Real-Time Latency and Quality:
The buffering approach introduces slight latency (controlled via --min_buffer). For smoother transitions, consider implementing overlapping windows and crossfading.
- Conversion Artifacts:
If the output sounds robotic, consider adjusting noise reduction parameters, blending your custom embedding with the model’s internal embedding, or experimenting with different f0 extraction methods.

## Acknowledgements

This project builds upon the so‑vits‑svc‑fork project. Many thanks to all contributors to that repository for their work on real-time voice conversion technologies.

---

Feel free to contribute, open issues, or share improvements. Enjoy converting voices in real time!
