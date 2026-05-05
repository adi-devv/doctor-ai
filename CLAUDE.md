# VedicAI — CLAUDE.md

## What this project is
VedicAI is an Ayurveda + yoga AI health assistant delivered as a voice-first web app. Users speak in any of 9 Indian languages; the backend transcribes, runs an LLM consultation, and streams back translated + TTS audio sentence-by-sentence.

## Stack
| Layer | Tech |
|---|---|
| Backend | Python 3.9, Flask, Gunicorn |
| Frontend | Single-file vanilla JS + CSS (`static/index.html`) — no build step |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| STT | Sarvam AI `saarika:v2.5` |
| TTS | Sarvam AI `bulbul:v3` (speaker: ritu) |
| Translation | Sarvam AI `mayura:v1` |
| Auth + DB | Firebase Auth (Google sign-in) + Firestore (chat history) |
| Yoga images | Wikipedia REST API (cached in-process) |
| Deployment | Docker + Gunicorn; Procfile (`python app.py`) for local |

## Running locally
```
python app.py          # port 7860
# or
gunicorn app:app --bind 0.0.0.0:7860
```

## Environment variables (`.env`)
```
SARVAM_API_KEY=...
GROQ_API_KEY=...
```

## Key files
- `app.py` — entire backend (Flask routes, LLM, TTS, translation, session state)
- `static/index.html` — entire frontend (Firebase, UI, audio recording, SSE consumer)
- `Dockerfile` — Python 3.9 image, runs gunicorn
- `requirements.txt` — flask, flask-cors, groq, requests, python-dotenv, gunicorn

## Architecture

### Request flow (voice turn)
1. Browser records audio → `POST /transcribe` → Sarvam STT → transcript
2. `POST /chat_stream` (SSE) with transcript text
3. Backend translates user text → English via Sarvam translate
4. Emergency regex check (before LLM)
5. Groq streams sentences; each sentence is translated + TTS'd in parallel (`ThreadPoolExecutor` 8 workers)
6. SSE events streamed: `sentence` → `plan` (if triggered) → `exchange_en` → `done`

### Session state (`SESSIONS` dict, in-memory)
- Keyed by session ID from `X-Session-Id` header or `sid` cookie
- Fields: `lang_code`, `history` (LLM messages), `turn_count`, `user_profile`, `consulting_for`
- **`MAX_HISTORY_TURNS = 12`** — must stay in sync with frontend's `MAX_TURNS_PER_CHAT`

### SSE events (from `/chat_stream`)
| Event | Payload |
|---|---|
| `sentence` | `{text, audio (base64), poses?: [{name, image}], emergency?: true}` |
| `plan` | `{plan: {...JSON plan...}}` |
| `exchange_en` | `{user_en, reply_en}` — for Firestore dual-field save |
| `done` | `{}` |
| `error` | `{msg, text?}` |

### Plan generation
LLM appends `[GENERATE_PLAN]` on its own line on the first substantive advice turn. Backend strips the tag, then calls Groq again with `PLAN_SYSTEM_PROMPT` to produce a structured JSON recovery plan, emitted as the `plan` SSE event.

### Translation — English passthrough
Medical/brand terms (paracetamol, ORS, WhatsApp, etc.) are replaced with numeric placeholders before calling Sarvam translate, then restored afterward. Feminine Hindi gender fixes are applied post-translation.

## Routes
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve `index.html` |
| POST | `/set_language` | Init session + greeting TTS |
| GET | `/greeting` | Greeting text only (no TTS, used when restoring a past chat) |
| POST | `/restore_session` | Rebuild LLM history from saved Firestore messages |
| POST | `/change_language` | Switch lang mid-chat without resetting history |
| POST | `/transcribe` | STT via Sarvam |
| POST | `/chat_stream` | Main streaming chat (SSE) |
| POST | `/reset` | Clear session |
| GET | `/health` | `{ok, sessions}` |

## User profile fields
`name`, `age`, `gender`, `location`, `conditions`, `medications`, `allergies`, `dosha` (Pitta/Vata/Kapha — set by 7-question in-app quiz)

The `consulting_for` field allows consulting on behalf of a family member (has `name`, `relation`, `age`, `gender`, `conditions`). When set, the system prompt instructs the LLM to address the caller as caregiver.

## Supported languages
`hi-IN` (Hindi, default), `en-IN`, `bn-IN`, `pa-IN`, `gu-IN`, `ta-IN`, `te-IN`, `kn-IN`, `ml-IN`
Script auto-detection is used to handle mixed-script input.

## Safety rules baked into the system prompt
- Emergency regex (chest pain, stroke symptoms, high fever, etc.) → immediate hospital redirect before LLM
- Self-harm regex → iCall / Vandrevala helpline numbers
- No Rx drugs; only OTC (Paracetamol, ORS, Digene/Eno, Cetirizine) when Ayurvedic approach is insufficient
- Under-12 / over-65 / pregnancy / Pitta/Vata/Kapha safety contraindications per remedy
- LLM is guardrailed to recommend only from an explicit approved list

## CORS allowed origins
`https://vedic.web.app`, `https://vedic.firebaseapp.com`, `http://localhost:7860`, `http://127.0.0.1:7860`

## Firebase / Firestore
Chat history is saved client-side in Firestore. Each message stores both local-language `text` and English `text_en` (dual-field). `/restore_session` translates old messages back to English to reconstruct LLM context.

## Yoga pose images
~50 poses mapped in `POSE_WIKI_PAGES` (canonical name → Wikipedia page slug). Images fetched from Wikipedia REST API and cached in-process (`_POSE_IMAGE_CACHE`). Poses are detected by regex in the English LLM reply and attached to the `sentence` SSE event.
