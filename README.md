---
title: Departmental Chatbot API
emoji: 🚛
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Department Knowledge Assistant API

RAG-based PDF question answering system for Tata Motors vehicle
performance and fuel economy documents. FastAPI backend served on
Hugging Face Spaces (free CPU Basic tier).

## Endpoints
- `GET /` — status check
- `GET /health` — health check
- `POST /chat` — `{"question": "..."}` → answer + retrieved sources

## Setup notes
- Set `GROQ_API_KEY` as a Space **secret** (Settings → Repository
  secrets) — do not commit your `.env` file.
- Commit the `vectorstore/` folder built by `build_vector.py` so it
  ships inside the Docker image; if any file in it is larger than
  ~10MB, use Git LFS (`git lfs track "vectorstore/*"`).
