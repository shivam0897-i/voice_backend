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

Detects whether a voice sample is AI-generated or spoken by a real human using a fine-tuned Wav2Vec2 model.

## API Endpoint

`POST /api/voice-detection`

### Headers
- `x-api-key`: Your API key (set via environment variable `API_KEY`)

### Request Body
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-audio>"
}
```

### Response
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED" | "HUMAN",
  "confidenceScore": 0.95,
  "explanation": "AI voice indicators: ..."
}
```

## Supported Languages
- English
- Tamil
- Hindi
- Malayalam
- Telugu



## Realtime Session APIs

The backend also supports session-based realtime analysis:

- `POST /v1/session/start`
- `POST /v1/session/{session_id}/chunk`
- `GET /v1/session/{session_id}/summary`
- `GET /v1/session/{session_id}/alerts`
- `POST /v1/session/{session_id}/end`

Compatibility aliases are available under `/api/voice-detection/v1/...`.

## Optional LLM Semantic Verifier

A second-layer semantic verifier can be enabled to improve ambiguous chunk scoring:

- `LLM_SEMANTIC_ENABLED=true`
- `LLM_PROVIDER=openai` with `OPENAI_API_KEY=<your_key>`, or
- `LLM_PROVIDER=gemini` with `GEMINI_API_KEY=<your_key>`
- Tune with `LLM_SEMANTIC_*` env variables in `.env.example`.

If `LLM_SEMANTIC_MODEL` is empty, provider defaults are used (`gpt-4o-mini` for OpenAI, `gemini-1.5-flash` for Gemini).

The LLM layer is optional and the API continues to work when disabled.


## Session Store Backend

Realtime sessions support two backends:

- `memory` (default): single-instance, volatile
- `redis`: multi-worker and restart-safe (recommended for finals)

Backend env settings:

- `SESSION_STORE_BACKEND=redis`
- `REDIS_URL=redis://...` (or `rediss://...`)
- `REDIS_PREFIX=ai_call_shield`

`GET /health` now includes `session_store_backend` so you can verify active backend.

See `docs/architecture/redis-credentials-guide.md` for credential formats and setup steps.
