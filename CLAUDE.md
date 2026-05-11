# VedicAI — CLAUDE.md

## What this project is
VedicAI is an Ayurveda + yoga AI health assistant delivered as a voice-first web app. Users speak in any of 11 Indian languages; the backend transcribes, runs an LLM consultation, and streams back translated + TTS audio sentence-by-sentence. A "Go Live" mode streams audio continuously via Sarvam's real-time WebSocket STT (no tap-to-record).

## Stack
| Layer | Tech |
|---|---|
| Backend | Python 3.9, Flask + flask-sock, Gunicorn (gthread) |
| Frontend | Single-file vanilla JS + CSS (`static/index.html`) — no build step |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| STT (batch) | Sarvam AI `saarika:v2.5` via `POST /transcribe` |
| STT (live) | Sarvam AI `saaras:v3` — streaming WebSocket, `realtime_balanced` mode with VAD signals |
| TTS | Sarvam AI `bulbul:v3` (speaker: ritu) |
| Translation | Sarvam AI `mayura:v1` |
| Auth + DB | Firebase Auth (Google sign-in) + Firestore (chat history) |
| Yoga images | Wikipedia REST API (cached in-process) |
| Deployment | Docker + Gunicorn; `python app.py` for local; HF Space at `https://adi-devv-doctor-ai.hf.space` |

## Running locally
```
python app.py          # port 7860
# or
gunicorn app:app --bind 0.0.0.0:7860 --worker-class gthread --threads 4 --workers 1
```

## Environment variables (`.env`)
```
SARVAM_API_KEY=...
GROQ_API_KEY=...
```

## Key files
- `app.py` — entire backend (Flask routes, LLM, TTS, translation, session state, WS proxy)
- `static/index.html` — entire frontend (Firebase, UI, audio recording, SSE consumer, live STT)
- `Dockerfile` — Python 3.9 image, runs gunicorn with gthread worker
- `requirements.txt` — flask, flask-cors, flask-sock>=0.2, groq, requests, python-dotenv, websocket-client>=1.6, gunicorn

## Architecture

### Request flow — tap-to-speak mode
1. Browser records audio (MediaRecorder, auto-stops on silence) → `POST /transcribe` → Sarvam STT → transcript
2. `POST /chat_stream` (SSE) with transcript text
3. Backend translates user text → English via Sarvam translate
4. Emergency regex check (before LLM)
5. Groq streams sentences; each sentence is translated + TTS'd in parallel (`ThreadPoolExecutor` 8 workers)
6. SSE events streamed: `sentence` → `offer_plan` (if triggered) → `exchange_en` → `done`

### Request flow — Go Live mode (continuous voice)
1. Frontend opens `wss://<host>/ws/live` → backend proxies to `wss://api.sarvam.ai/speech-to-text-streaming`
2. AudioWorklet converts Float32 mic input to Int16 PCM and sends binary frames over WS
3. Sarvam emits `events` (VAD: `START_SPEECH` / `END_SPEECH`) and transcript data
4. On `END_SPEECH`: frontend flushes, waits for final transcript, calls `streamChat()` (same SSE flow)
5. WS stays open between turns — only reconnects if server closes it
6. Mic + AudioContext are set up once per live session; WS reconnects per turn if needed

### Session state (`SESSIONS` dict, in-memory)
- Keyed by session ID from `X-Session-Id` header or `sid` cookie
- Fields: `lang_code`, `history` (LLM messages), `turn_count`, `user_profile`, `consulting_for`
- **`MAX_HISTORY_TURNS = 12`** — must stay in sync with frontend's `MAX_TURNS_PER_CHAT`

### Greeting cache
- **Server-side**: `_warm_greeting_cache()` runs in a daemon thread at startup, pre-translating + TTS-ing the greeting in all 9 supported languages. Stored in `_GREETING_CACHE` dict.
- **Client-side**: `localStorage` stores `vedic_greet_<lang_code>` (text) and `vedic_greet_audios_<lang_code>` (base64 audio array). On language chip click, cached greeting is shown and played instantly while `/set_language` runs in the background.

### SSE events (from `/chat_stream`)
| Event | Payload |
|---|---|
| `sentence` | `{text, audio (base64), poses?: [{name, image}], emergency?: true}` |
| `offer_plan` | `{}` — client shows opt-in UI |
| `exchange_en` | `{user_en, reply_en}` — for Firestore dual-field save |
| `done` | `{}` |
| `error` | `{msg, text?}` |

### Plan generation (opt-in)
LLM appends `[OFFER_PLAN]` on the first substantive advice turn. Backend strips the tag and emits an `offer_plan` SSE event. The frontend shows a non-intrusive yes/no prompt. If user accepts, the frontend calls `POST /generate_plan` which calls Groq with `PLAN_SYSTEM_PROMPT` and returns structured JSON rendered as a plan card.

### Translation — English passthrough
Medical/brand terms (paracetamol, ORS, WhatsApp, etc.) are replaced with numeric placeholders before calling Sarvam translate, then restored afterward. Feminine Hindi gender fixes are applied post-translation.

## Routes
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve `index.html` |
| POST | `/set_language` | Init session + greeting TTS (uses cache if warm) |
| GET | `/greeting` | Greeting text only (no TTS, used when restoring a past chat) |
| POST | `/restore_session` | Rebuild LLM history from saved Firestore messages |
| POST | `/change_language` | Switch lang mid-chat without resetting history |
| POST | `/transcribe` | Batch STT via Sarvam saarika:v2.5 |
| POST | `/chat_stream` | Main streaming chat (SSE) |
| POST | `/generate_plan` | On-demand recovery plan JSON (user opted in) |
| POST | `/reset` | Clear session |
| GET | `/health` | `{ok, sessions}` |
| WS | `/ws/live` | WebSocket proxy → Sarvam saaras:v3 live STT |

## User profile fields
`name`, `age`, `gender`, `location`, `conditions`, `medications`, `allergies`, `dosha` (Pitta/Vata/Kapha — set by 7-question in-app quiz)

The `consulting_for` field allows consulting on behalf of a family member (has `name`, `relation`, `age`, `gender`, `conditions`). When set, the system prompt instructs the LLM to address the caller as caregiver.

## Supported languages
Frontend chips: `en-IN`, `hi-IN`, `ta-IN`, `te-IN`, `mr-IN`, `bn-IN`, `kn-IN`, `gu-IN`, `ml-IN`, `pa-IN`, `od-IN`
Server-side greeting cache covers 9: `hi-IN`, `en-IN`, `bn-IN`, `pa-IN`, `gu-IN`, `ta-IN`, `te-IN`, `kn-IN`, `ml-IN`
Script auto-detection used for restore-session and en-IN passthrough handling.

## Safety rules baked into the system prompt
- Emergency regex (chest pain, stroke symptoms, high fever, etc.) → immediate hospital redirect before LLM
- Self-harm regex → iCall / Vandrevala helpline numbers
- No Rx drugs; only OTC (Paracetamol, ORS, Digene/Eno, Cetirizine) when Ayurvedic approach is insufficient
- Under-12 / over-65 / pregnancy / Pitta/Vata/Kapha safety contraindications per remedy
- LLM is guardrailed to recommend only from an explicit approved list

## CORS allowed origins
`https://vedic.web.app`, `https://vedic.firebaseapp.com`, `http://localhost:7860`, `http://127.0.0.1:7860`
(HF Space origin is same-origin when serving its own frontend — no CORS needed for that case)

## Firebase / Firestore
Chat history is saved client-side in Firestore. Each message stores both local-language `text` and English `text_en` (dual-field). `/restore_session` translates old messages back to English to reconstruct LLM context.

## Yoga pose images
~55 poses mapped in `POSE_WIKI_PAGES` (canonical name → Wikipedia page slug). Images fetched from Wikipedia REST API and cached in-process (`_POSE_IMAGE_CACHE`). Poses are detected by regex in the English LLM reply and attached to the `sentence` SSE event.

## Live mode key behaviours
- Entering live mode while greeting audio is playing: defers mic/WS setup until audio finishes (prevents audio interruption from `getUserMedia`)
- WS keepalive: backend pings Sarvam every 15s to keep the streaming connection alive
- Between turns: WS stays open; `liveSendingAudio` gate controls whether PCM reaches Sarvam
- On WS close mid-idle: auto-reconnects after 400ms if not mid-turn
