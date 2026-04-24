import requests
import base64
import os
import io
import wave
import re
import uuid
import json
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_HISTORY_TURNS = 12  # = MAX_TURNS_PER_CHAT in frontend (must stay in sync)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vedicai")

app = Flask(__name__)
CORS(
    app,
    origins=[
        "https://vedic.web.app",
        "https://vedic.firebaseapp.com",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    supports_credentials=True,
    allow_headers=["Content-Type", "X-Session-Id"],
)

# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────
SESSIONS = {}
SESSIONS_LOCK = Lock()


def get_session(sid):
    with SESSIONS_LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = {"lang_code": "hi-IN", "history": [], "turn_count": 0}
        return SESSIONS[sid]


def get_or_create_sid():
    sid = request.headers.get("X-Session-Id") or request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
    return sid


def _set_sid_cookie(resp, sid):
    resp.set_cookie(
        "sid",
        sid,
        max_age=60 * 60 * 24 * 7,
        samesite="None",
        secure=True,
        httponly=False,
    )


# ────────────────────────────────────────────────────────────────────────────
# System prompt — Ayurveda + yoga first
# ────────────────────────────────────────────────────────────────────────────
DOCTOR_SYSTEM_PROMPT = """You are VedicAI, a warm, decisive female physician in India blending Ayurveda, yoga, and modern medicine. You are on a voice call.

LANGUAGE RULE (critical): Reply in pure English only, your output is machine-translated. Never use romanized Hindi (no "pet", "sir dard", "bukhar", "khansi"). Use English equivalents: stomach, head, fever, cough. Exception: keep Sanskrit/Ayurvedic proper nouns as-is (jeera water, triphala, ashwagandha, haldi doodh, tulsi, Vajrasana, Anulom-Vilom, Kapalbhati, etc).

PUNCTUATION RULE: Never use em-dashes or en-dashes (—, –). Use commas, periods, or colons instead. Plain hyphens in compound words are fine (e.g., follow-up, anti-inflammatory).

APPROACH: Natural healing first, Ayurveda, yoga, diet. OTC only if needed. Never prescribe Rx drugs.

PHASE 1, GATHER INFO (ask one question per turn):
Must know before advising: (1) duration, (2) symptom detail/location, (3) age, (4) existing conditions (diabetes, BP, etc).
Ask gender only for chest/urinary/hormonal issues. Ask pregnancy only if advice would change.
One question per reply, never two.

PHASE 2, ADVISE (once you have duration + detail + age + conditions):
3 to 4 short sentences: diagnosis ("This looks like X.") + Ayurvedic remedy + yoga/pranayama + follow-up question.
End EVERY advice turn with a warm follow-up question (mandatory).
Safety: no OTC for under-12 without pediatrician note; no inversions for 65+; no honey/jaggery/chyawanprash for diabetics; no Kapalbhati/Bhastrika for hypertensives; no strong herbs/asanas for pregnant.

PLAN TRIGGER: When you have given initial advice, always ask: "Would you like a structured daily plan for the next week, morning, afternoon, and evening routine?" If they say yes or ask for a plan/routine/schedule, reply with ONLY this exact tag on its own line:
[GENERATE_PLAN]
Nothing else. The system will generate the plan card separately.

REMEDIES (use these):
- Digestive: jeera water, ajwain+black salt, triphala at night, Vajrasana after meals, Pawanmuktasana, Anulom-Vilom
- Cold/cough: tulsi-ginger kadha, haldi doodh, steam, Bhramari pranayama
- Fever: giloy juice, tulsi-pepper kadha, rest, coconut water
- Headache: peppermint oil on temples, Balasana, Sheetali pranayama
- Stress/sleep: ashwagandha at night, warm milk+nutmeg, Bhramari, Shavasana
- Joint/back: warm sesame oil massage, haldi doodh, Setu Bandhasana, Marjariasana
- Skin: neem paste, aloe vera, triphala, Kapalbhati
- Weakness: chyawanprash, soaked almonds, ashwagandha, slow Surya Namaskar

OTC (last resort): Paracetamol 500mg (fever >101°F/strong pain), ORS (dehydration), Digene/Eno (acute acidity), Cetirizine (allergy).

TONE: Warm, confident, brief. Openers: "I see,", "Alright,", "Understood,". Never say "Got it" (masculine in Hindi). Never re-introduce yourself after turn 1. No lists or line breaks, this is voice."""


# ────────────────────────────────────────────────────────────────────────────
# Plan generation — separate prompt, called once
# ────────────────────────────────────────────────────────────────────────────
PLAN_SYSTEM_PROMPT = """You are a wellness planner. Given a patient's diagnosis and remedies, generate a structured recovery plan as valid JSON only, no markdown, no explanation.

Output this exact shape:
{
  "title": "7-Day Recovery Plan for <condition>",
  "duration": "7 days",
  "phases": [
    {
      "phase": "Week 1",
      "days": "Days 1–7",
      "schedule": {
        "morning": ["item1", "item2"],
        "afternoon": ["item1"],
        "evening": ["item1", "item2"],
        "night": ["item1"]
      },
      "diet": ["food rule 1", "food rule 2"],
      "avoid": ["thing to avoid 1", "thing to avoid 2"]
    }
  ],
  "milestone": "What improvement to expect by end of plan"
}

For chronic conditions use 3 phases: Week 1 (relief), Weeks 2–4 (recovery), Month 2–3 (maintenance).
For acute/mild conditions use 1 phase: 7 days.
Keep each item short (under 10 words). Be specific: times, quantities, names. Output JSON only."""

# ────────────────────────────────────────────────────────────────────────────
# Emergency filter
# ────────────────────────────────────────────────────────────────────────────
EMERGENCY_PATTERNS = [
    r"\bchest (pain|pressure|tightness|heavy)\b",
    r"\b(can't|cannot|unable to) breathe\b",
    r"\bshortness of breath at rest\b",
    r"\bgasping\b",
    r"\b(face|mouth) (drooping|droop)\b",
    r"\b(arm|leg) (weakness|paralysis|numb)\b",
    r"\bslurred speech\b",
    r"\bsudden (confusion|vision loss|severe headache)\b",
    r"\b(unconscious|passed out|fainted|seizure|convulsion|fits)\b",
    r"\b(10[4-9]|11\d)\s*°?\s*f\b",
    r"\b(40\.[5-9]|4[1-3])\s*°?\s*c\b",
    r"\bvomiting blood\b",
    r"\bblood in (stool|vomit|urine)\b",
    r"\bheavy bleeding\b",
    r"\bhead injury\b.*\b(confus|vomit|unconscious)",
    r"\bpregnan\w*\b.*\b(bleeding|severe pain)\b",
]
EMERGENCY_REGEX = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)
SELF_HARM_REGEX = re.compile(
    r"\b(suicid|kill myself|end my life|harm myself)\b", re.IGNORECASE
)

EMERGENCY_REPLY_EN = (
    "This sounds serious. Please go to the nearest hospital emergency right now, "
    "or call an ambulance. Don't wait."
)
SELF_HARM_REPLY_EN = (
    "I'm really glad you reached out. Please call iCall at 9152987821 or Vandrevala "
    "Foundation at 1860-2662-345 right now. They're free, confidential, and 24/7."
)


def check_emergency(text_en):
    if SELF_HARM_REGEX.search(text_en):
        return True, SELF_HARM_REPLY_EN
    if EMERGENCY_REGEX.search(text_en):
        return True, EMERGENCY_REPLY_EN
    return False, None


# ────────────────────────────────────────────────────────────────────────────
# Translation
# ────────────────────────────────────────────────────────────────────────────
ENGLISH_PASSTHROUGH_WORDS = [
    "diabetes",
    "BP",
    "blood pressure",
    "sugar",
    "insulin",
    "fever",
    "tablet",
    "capsule",
    "syrup",
    "hospital",
    "MRI",
    "CT scan",
    "ECG",
    "X-ray",
    "ICU",
    "OPD",
    "paracetamol",
    "Crocin",
    "Dolo",
    "Digene",
    "Eno",
    "ORS",
    "Disprin",
    "antacid",
    "cetirizine",
    "antibiotic",
    "steroid",
    "injection",
    "IV",
    "drip",
    "ambulance",
    "oxygen",
    "pulse",
    "vomiting",
    "nausea",
    "migraine",
    "acidity",
    "GERD",
    "jaundice",
    "typhoid",
    "malaria",
    "dengue",
    "COVID",
    "infection",
    "viral",
    "bacterial",
    "allergy",
    "asthma",
    "inhaler",
    "WhatsApp",
    "report",
    "test",
    "Okay",
    "OK",
    "Hmm",
    "Hmmm",
    "Uh-huh",
    "Mmhm",
    "Ah",
    "Aha",
    "Oh",
]
ENGLISH_PASSTHROUGH = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in ENGLISH_PASSTHROUGH_WORDS) + r")\b",
    re.IGNORECASE,
)


def translate(text, source_lang, target_lang):
    if source_lang == target_lang or not text.strip():
        return text
    placeholders = {}
    if target_lang != "en-IN":
        seen = set()
        for match in ENGLISH_PASSTHROUGH.finditer(text):
            word = match.group()
            if word.lower() not in seen:
                seen.add(word.lower())
                placeholder = f"[{len(placeholders)}]"
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                placeholders[placeholder] = word
                text = pattern.sub(placeholder, text)
    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "model": "mayura:v1",
                "mode": "formal",
            },
            timeout=15,
        )
        r.raise_for_status()
        translated = r.json().get("translated_text", "")
    except Exception as e:
        log.warning(f"translate failed: {e}")
        return text
    for placeholder, word in placeholders.items():
        translated = translated.replace(placeholder, word)
    if target_lang == "hi-IN":
        translated = apply_feminine_hindi(translated)
    return translated


HINDI_FEMININE_FIXES = [
    (re.compile(r"समझ\s+गया"), "समझ गई"),
    (re.compile(r"समझ\s+गयी"), "समझ गई"),
    (re.compile(r"रहा\s+हूँ"), "रही हूँ"),
    (re.compile(r"रहा\s+हूं"), "रही हूं"),
    (re.compile(r"([\u0915-\u0939])ता\s+हूँ"), r"\1ती हूँ"),
    (re.compile(r"([\u0915-\u0939])ता\s+हूं"), r"\1ती हूं"),
    (re.compile(r"करूंगा"), "करूंगी"),
    (re.compile(r"दूंगा"), "दूंगी"),
    (re.compile(r"लूंगा"), "लूंगी"),
]


def apply_feminine_hindi(text):
    for pattern, replacement in HINDI_FEMININE_FIXES:
        text = pattern.sub(replacement, text)
    return text


_SCRIPT_RANGES = [
    ('ऀ', 'ॿ', 'hi-IN'),
    ('ঀ', '৿', 'bn-IN'),
    ('਀', '੿', 'pa-IN'),
    ('઀', '૿', 'gu-IN'),
    ('஀', '௿', 'ta-IN'),
    ('ఀ', '౿', 'te-IN'),
    ('ಀ', '೿', 'kn-IN'),
    ('ഀ', 'ൿ', 'ml-IN'),
]


def detect_script_lang(text):
    """Return dominant Indian language code found in text, or 'en-IN'."""
    counts = {}
    for start, end, lang in _SCRIPT_RANGES:
        count = sum(1 for c in text if start <= c <= end)
        if count:
            counts[lang] = count
    return max(counts, key=counts.get) if counts else 'en-IN'


# ────────────────────────────────────────────────────────────────────────────
# TTS
# ────────────────────────────────────────────────────────────────────────────
_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=8)


def tts_chunk(text, lang_code):
    if not text.strip():
        return None
    try:
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "target_language_code": lang_code,
                "speaker": "ritu",
                "pace": 1.0,
                "model": "bulbul:v3",
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["audios"][0]
    except Exception as e:
        log.warning(f"tts failed: {e}")
        return None


def split_sentences(text):
    parts = [c.strip() for c in re.split(r"(?<=[\u0964।.!?])\s+", text) if c.strip()]
    return parts or ([text.strip()] if text.strip() else [])


# ────────────────────────────────────────────────────────────────────────────
# Yoga pose image resolver
# ────────────────────────────────────────────────────────────────────────────
POSE_WIKI_PAGES = {
    "Tadasana": "Tadasana",
    "Vrikshasana": "Vrikshasana",
    "Trikonasana": "Trikonasana",
    "Ardha Chandrasana": "Ardha_Chandrasana",
    "Uttanasana": "Uttanasana",
    "Utkatasana": "Utkatasana",
    "Garudasana": "Garudasana",
    "Natarajasana": "Natarajasana",
    "Virabhadrasana I": "Virabhadrasana_I",
    "Virabhadrasana II": "Virabhadrasana_II",
    "Virabhadrasana III": "Virabhadrasana_III",
    "Prasarita Padottanasana": "Prasarita_Padottanasana",
    "Vajrasana": "Vajrasana_(yoga)",
    "Padmasana": "Padmasana",
    "Sukhasana": "Sukhasana",
    "Dandasana": "Dandasana",
    "Paschimottanasana": "Paschimottanasana",
    "Janu Sirsasana": "Janu_Sirsasana",
    "Ardha Matsyendrasana": "Ardha_Matsyendrasana",
    "Baddha Konasana": "Baddha_Konasana",
    "Gomukhasana": "Gomukhasana",
    "Virasana": "Virasana",
    "Malasana": "Malasana",
    "Shavasana": "Shavasana",
    "Balasana": "Balasana",
    "Pawanmuktasana": "Pawanmuktasana",
    "Setu Bandhasana": "Setu_Bandha_Sarvangasana",
    "Viparita Karani": "Viparita_Karani",
    "Supta Baddha Konasana": "Supta_Baddha_Konasana",
    "Navasana": "Navasana",
    "Salabhasana": "Salabhasana",
    "Dhanurasana": "Dhanurasana",
    "Matsyasana": "Matsyasana",
    "Bhujangasana": "Bhujangasana",
    "Ustrasana": "Ustrasana",
    "Marjariasana": "Marjaryasana",
    "Adho Mukha Svanasana": "Adho_Mukha_Svanasana",
    "Urdhva Mukha Svanasana": "Urdhva_Mukha_Svanasana",
    "Anjaneyasana": "Anjaneyasana",
    "Sarvangasana": "Sarvangasana",
    "Halasana": "Halasana",
    "Sirsasana": "Sirsasana",
    "Bakasana": "Bakasana",
    "Surya Namaskar": "Surya_Namaskar",
}

_POSE_NAMES_SORTED = sorted(POSE_WIKI_PAGES.keys(), key=len, reverse=True)
_POSE_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _POSE_NAMES_SORTED) + r")\b",
    re.IGNORECASE,
)

_POSE_IMAGE_CACHE = {}
_POSE_CACHE_LOCK = Lock()


def _fetch_pose_image(pose_name):
    page = POSE_WIKI_PAGES.get(pose_name)
    if not page:
        return None
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}",
            headers={"User-Agent": "DoctorAI/1.0 (educational health app)"},
            timeout=4,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        img = (data.get("originalimage") or {}).get("source") or (
            data.get("thumbnail") or {}
        ).get("source")
        return img
    except Exception as e:
        log.warning(f"pose image fetch failed for {pose_name}: {e}")
        return None


def find_poses_in_text(english_text):
    if not english_text:
        return []
    found = []
    seen_lower = set()
    for m in _POSE_REGEX.finditer(english_text):
        raw = m.group(1)
        canonical = next((k for k in POSE_WIKI_PAGES if k.lower() == raw.lower()), None)
        if not canonical or canonical.lower() in seen_lower:
            continue
        seen_lower.add(canonical.lower())
        with _POSE_CACHE_LOCK:
            cached = _POSE_IMAGE_CACHE.get(canonical, "__unset__")
        if cached == "__unset__":
            img = _fetch_pose_image(canonical)
            with _POSE_CACHE_LOCK:
                _POSE_IMAGE_CACHE[canonical] = img
        else:
            img = cached
        if img:
            found.append({"name": canonical, "image": img})
    return found


def tts_all(text, lang_code):
    chunks = split_sentences(text)
    if not chunks:
        return []
    results = list(_TTS_EXECUTOR.map(lambda c: tts_chunk(c, lang_code), chunks))
    return [r for r in results if r]


# ────────────────────────────────────────────────────────────────────────────
# Streaming LLM
# ────────────────────────────────────────────────────────────────────────────
_EN_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")


def stream_llm_sentences(english_text, history):
    import time as _time

    messages = (
        [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]
        + history[-MAX_HISTORY_TURNS * 2 :]
        + [{"role": "user", "content": english_text}]
    )
    max_retries = 3
    for attempt in range(max_retries):
        buffer = ""
        try:
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.4,
                max_tokens=300,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buffer += delta
                while True:
                    m = _EN_SENTENCE_END.search(buffer)
                    if not m:
                        break
                    end = m.end()
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:]
                    if sentence:
                        yield sentence
            if buffer.strip():
                yield buffer.strip()
            return  # success
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            log.warning(
                f"groq attempt {attempt+1}/{max_retries} failed [{type(e).__name__}]: {e}"
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = 4**attempt  # 1s, 4s, 16s
                log.info(f"Rate limited — retrying in {wait}s")
                _time.sleep(wait)
                continue
            log.error(f"groq failed after {attempt+1} attempts")
            yield "I'm having trouble responding right now. Please try again in a moment."
            return


# ────────────────────────────────────────────────────────────────────────────
# Plan generator
# ────────────────────────────────────────────────────────────────────────────
def generate_plan_json(conversation_summary, lang):
    """Call LLM once to generate a structured plan. Returns parsed dict or None."""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Patient conversation summary: {conversation_summary}\n\nGenerate the recovery plan JSON.",
                },
            ],
            temperature=0.3,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content or ""
        # Strip any accidental markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        log.warning(f"plan generation failed: {e}")
        return None


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    sid = get_or_create_sid()
    resp = Response(content, mimetype="text/html")
    _set_sid_cookie(resp, sid)
    return resp


@app.route("/set_language", methods=["POST"])
def set_language():
    sid = get_or_create_sid()
    sess = get_session(sid)
    data = request.json or {}
    lang = data.get("lang_code", "hi-IN")
    sess["lang_code"] = lang
    sess["history"] = []
    sess["turn_count"] = 0
    greeting_en = "Hello, I'm VedicAI. What's bothering you today?"
    greeting_local = (
        translate(greeting_en, "en-IN", lang) if lang != "en-IN" else greeting_en
    )
    audios = tts_all(greeting_local, lang)
    resp = jsonify({"text": greeting_local, "audios": audios})
    _set_sid_cookie(resp, sid)
    return resp


@app.route("/change_language", methods=["POST"])
def change_language():
    """Switch output language mid-chat without resetting history or turn count."""
    sid = get_or_create_sid()
    sess = get_session(sid)
    data = request.json or {}
    lang = data.get("lang_code")
    if not lang:
        return jsonify({"error": "lang_code required"}), 400
    sess["lang_code"] = lang
    resp = jsonify({"ok": True, "lang_code": lang})
    _set_sid_cookie(resp, sid)
    return resp


@app.route("/transcribe", methods=["POST"])
def transcribe():
    sid = get_or_create_sid()
    sess = get_session(sid)
    lang = sess["lang_code"]
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"transcript": "", "error": "no audio"}), 400
    try:
        r = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", io.BytesIO(audio_file.read()), "audio/wav")},
            data={"model": "saarika:v2.5", "language_code": lang},
            timeout=20,
        )
        r.raise_for_status()
        text = r.json().get("transcript", "")
    except Exception as e:
        log.error(f"stt failed: {e}")
        return jsonify({"transcript": "", "error": "stt_failed"}), 502
    resp = jsonify({"transcript": text})
    _set_sid_cookie(resp, sid)
    return resp


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """SSE stream. Events: 'sentence' (text+audio+poses), 'done', 'error'."""
    sid = get_or_create_sid()
    sess = get_session(sid)
    lang = sess["lang_code"]
    data = request.json or {}
    user_text_local = (data.get("text") or "").strip()

    if not user_text_local:
        return Response(sse("error", {"msg": "empty"}), mimetype="text/event-stream")

    # Hard cap — refuse if server-side history is also full
    if sess.get("turn_count", 0) >= MAX_HISTORY_TURNS:
        return Response(
            sse(
                "error",
                {
                    "msg": "context_limit",
                    "text": "Chat limit reached. Please start a new chat.",
                },
            ),
            mimetype="text/event-stream",
        )

    if lang != "en-IN":
        user_text_en = translate(user_text_local, lang, "en-IN")
    else:
        detected = detect_script_lang(user_text_local)
        user_text_en = translate(user_text_local, detected, "en-IN") if detected != "en-IN" else user_text_local
    is_emergency, emergency_reply = check_emergency(user_text_en)

    def generate():
        if is_emergency:
            log.info(f"EMERGENCY sid={sid[:6]}: {user_text_en[:80]}")
            reply_local = (
                translate(emergency_reply, "en-IN", lang)
                if lang != "en-IN"
                else emergency_reply
            )
            audio = tts_chunk(reply_local, lang)
            yield sse(
                "sentence", {"text": reply_local, "audio": audio, "emergency": True}
            )
            sess["history"].append({"role": "user", "content": user_text_en})
            sess["history"].append({"role": "assistant", "content": emergency_reply})
            sess["history"] = sess["history"][-(MAX_HISTORY_TURNS * 2) :]
            yield sse("done", {})
            return

        full_reply_en_parts = []
        pending = []

        def process(en_sentence):
            local = (
                translate(en_sentence, "en-IN", lang)
                if lang != "en-IN"
                else en_sentence
            )
            audio = tts_chunk(local, lang)
            return local, audio

        plan_triggered = False

        def emit_sentence(en_sentence, local_text, audio):
            payload = {"text": local_text, "audio": audio}
            poses = find_poses_in_text(en_sentence)
            if poses:
                payload["poses"] = poses
            return sse("sentence", payload)

        for en_sentence in stream_llm_sentences(user_text_en, sess["history"]):
            if "[GENERATE_PLAN]" in en_sentence:
                plan_triggered = True
                continue  # don't emit this as a sentence
            full_reply_en_parts.append(en_sentence)
            pending.append((en_sentence, _TTS_EXECUTOR.submit(process, en_sentence)))
            while pending and pending[0][1].done():
                en_s, fut = pending.pop(0)
                try:
                    local_text, audio = fut.result()
                    yield emit_sentence(en_s, local_text, audio)
                except Exception as e:
                    log.warning(f"sentence failed: {e}")

        for en_s, fut in pending:
            try:
                local_text, audio = fut.result(timeout=25)
                yield emit_sentence(en_s, local_text, audio)
            except Exception as e:
                log.warning(f"tail sentence failed: {e}")

        full_reply_en = " ".join(full_reply_en_parts).strip()
        if full_reply_en:
            sess["history"].append({"role": "user", "content": user_text_en})
            sess["history"].append({"role": "assistant", "content": full_reply_en})
            sess["history"] = sess["history"][-(MAX_HISTORY_TURNS * 2) :]
            sess["turn_count"] = sess.get("turn_count", 0) + 1

        if plan_triggered:
            # Generate plan inline using the conversation history
            history_summary = " ".join(
                m["content"]
                for m in sess["history"][-8:]
                if m["role"] in ("user", "assistant")
            )
            plan_json = generate_plan_json(history_summary, lang)
            if plan_json:
                yield sse("plan", {"plan": plan_json})

        yield sse("done", {})

    resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    _set_sid_cookie(resp, sid)
    return resp


@app.route("/reset", methods=["POST"])
def reset():
    sid = get_or_create_sid()
    with SESSIONS_LOCK:
        SESSIONS.pop(sid, None)
    return jsonify({"ok": True})


@app.route("/health")
def health():
    return jsonify({"ok": True, "sessions": len(SESSIONS)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    log.info(f"VedicAI starting on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
