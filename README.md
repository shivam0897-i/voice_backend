---
title: Voice Detection API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# AI Voice Detection API

Detects whether a voice sample is **AI-generated** or spoken by a **real human** using a fine-tuned Wav2Vec2 transformer model combined with multi-signal forensic analysis.

## Model Architecture

```
Audio Input (Base64 MP3/WAV)
        │
        ▼
┌─────────────────────┐
│  Audio Preprocessing │  librosa 16 kHz mono, normalization
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────┐
│Wav2Vec2│ │ Signal Forensics │
│  Model │ │  (4 dimensions)  │
└───┬────┘ └───────┬──────────┘
    │              │
    ▼              ▼
  Softmax    ┌─────────────┐
 Confidence  │ Pitch       │
    │        │ Spectral    │
    │        │ Temporal    │
    │        │ Authenticity│
    │        └──────┬──────┘
    └───────┬───────┘
            ▼
   Final Classification
   (HUMAN / AI_GENERATED)
```

### Key Components

| Component | Description |
|-----------|-------------|
| **ML Backbone** | [Wav2Vec2ForSequenceClassification](https://huggingface.co/shivam-2211/voice-detection-model) fine-tuned on human vs. AI-generated speech |
| **Temperature Scaling** | Logits scaled by T=1.5 before softmax for well-calibrated confidence scores |
| **Signal Forensics** | Pitch stability, spectral entropy, temporal rhythm, and acoustic anomaly detection |
| **ASR Integration** | Faster-Whisper (tiny, int8) for language detection and transcript extraction |
| **Timeout Safety** | 20-second budget with audio truncation to guarantee <30s response |

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (`apt-get install ffmpeg` or `brew install ffmpeg`)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/shivam0897-i/voice_backend.git
cd voice_backend

# Install CPU-only PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "API_KEY=your_secret_key" > .env

# Run the server
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t voice-detection-api .
docker run -p 7860:7860 -e API_KEY=your_secret_key voice-detection-api
```

## API Endpoint

### `POST /api/voice-detection`

**Headers:**
| Header | Description |
|--------|-------------|
| `Content-Type` | `application/json` |
| `x-api-key` | Your API key (set via `API_KEY` env var) |

**Request Body:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-audio>"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.99,
  "explanation": "AI voice indicators detected with high confidence..."
}
```

**Example with curl:**
```bash
# Encode audio to Base64 and send
AUDIO_B64=$(base64 -w0 sample.mp3)
curl -X POST https://shivam-2211-voice-detection-api.hf.space/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d "{\"language\": \"English\", \"audioFormat\": \"mp3\", \"audioBase64\": \"$AUDIO_B64\"}"
```

## Supported Languages

| Language | Code |
|----------|------|
| English | `English` |
| Hindi | `Hindi` |
| Tamil | `Tamil` |
| Malayalam | `Malayalam` |
| Telugu | `Telugu` |
| Auto-detect | `Auto` |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | **Yes** | — | API authentication key |
| `MODEL_NAME` | No | `shivam-2211/voice-detection-model` | HuggingFace model ID |
| `MODEL_LOGIT_TEMPERATURE` | No | `1.5` | Softmax temperature scaling |
| `SESSION_STORE_BACKEND` | No | `redis` | Session backend (`memory` or `redis`) |
| `REDIS_URL` | No | — | Redis connection URL |
| `LLM_SEMANTIC_ENABLED` | No | `false` | Enable LLM semantic verifier |
| `PORT` | No | `7860` | Server port |

## Deployment

The API is deployed on **HuggingFace Spaces** using Docker:

- **Live URL**: `https://shivam-2211-voice-detection-api.hf.space`
- **Health Check**: `GET /health`
- **Infrastructure**: CPU inference, 2 Uvicorn workers, Redis session store

## Project Structure

```
├── main.py              # FastAPI app, all endpoints, error handling
├── model.py             # Wav2Vec2 inference + signal forensics engine
├── audio_utils.py       # Base64 decoding, audio validation, loading
├── config.py            # Pydantic Settings (env-based configuration)
├── speech_to_text.py    # Faster-Whisper ASR integration
├── fraud_language.py    # Fraud language pattern detection
├── privacy_utils.py     # PII redaction utilities
├── Dockerfile           # Production Docker image
├── requirements.txt     # Python dependencies
└── tests/               # Test suite
```

## License

MIT
