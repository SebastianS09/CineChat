# CineChat

Voice assistant that tells you what's playing nearby and recommends films.

## Stack
- **Backend**: FastAPI + [Gradium](https://gradium.ai) voice SDK (gradbot)
- **Movie data**: [bobine.art](https://bobine.art) showtimes API
- **LLM**: OpenAI-compatible (configurable)
- **Phone**: Twilio Media Streams (µ-law 8kHz on the wire, OggOpus internally)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.yaml config.local.yaml  # fill in API keys
uvicorn main:app --reload
```

## Test (mic → bot)
```bash
python test_ws_mic.py
```

## Phone (Twilio)
Point your Twilio number's webhook at `https://<host>/twilio/voice`.
