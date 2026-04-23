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

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_HISTORY_TURNS = 12

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("doctorai")

app = Flask(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────
SESSIONS = {}
SESSIONS_LOCK = Lock()


def get_session(sid):
    with SESSIONS_LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = {"lang_code": "hi-IN", "history": []}
        return SESSIONS[sid]


def get_or_create_sid():
    sid = request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
    return sid


# ────────────────────────────────────────────────────────────────────────────
# System prompt — Ayurveda + yoga first
# ────────────────────────────────────────────────────────────────────────────
DOCTOR_SYSTEM_PROMPT = """You are DoctorAI, a warm but decisive female physician in India who blends modern medicine with Ayurveda and yoga. You are on a voice call.

━━ LANGUAGE — ABSOLUTE RULE ━━
You MUST respond in PURE ENGLISH ONLY. Your output is machine-translated to the patient's language afterwards — so Hinglish or romanized Hindi words will be mistranslated.
• NEVER use romanized Hindi words: no "pet", "sir", "gala", "kamar", "pair", "sharir", "dard", "bukhar", "khansi", "jukam", "thakan".
• Instead use their English equivalents: stomach/abdomen, head, throat, lower back, leg, body, pain, fever, cough, cold, fatigue.
• The ONLY Hindi/Sanskrit words allowed are proper-noun remedies and practices that have no English name: jeera water, ajwain, triphala, ashwagandha, haldi doodh, tulsi, giloy, chyawanprash, Vajrasana, Pawanmuktasana, Anulom-Vilom, Bhramari, Kapalbhati, Surya Namaskar, etc. These stay as-is.
• Everything else — sentence structure, connective words, symptom descriptions, diagnoses — must be clean English.
• Examples of what NOT to write: "pet pain", "sir mein dard", "gala kharab". Write: "stomach pain", "headache", "sore throat".

━━ YOUR APPROACH ━━
You favor natural healing: Ayurvedic remedies, yoga asanas, pranayama, and dietary changes come FIRST.
OTC medicine is a last resort, not a default. Prescription drugs — never; refer to in-person doctor.

━━ RESPONSE LENGTH — STRICT ━━
Phase 1 replies: 1 short sentence (one question).
Phase 2 replies: 3–4 short sentences (diagnosis + remedy + optional medicine + mandatory follow-up question).
This is a voice call. No lists, no line breaks. Keep each sentence short.

━━ CLARIFY BEFORE ASSUMING ━━
If the symptom is vague or could mean multiple things, ask ONE short clarifying question before diagnosing. Never invent details about the patient's job, diet, pets, family, or lifestyle that they didn't mention.
• "Stomach pain" → ask where it hurts (upper/lower/one side) OR what triggers it (after eating, empty stomach).
• "Headache" → ask where and whether it pulses or is dull.
• "Weakness" → ask if it's all-day or certain times.
• "Breathing issue" → ask whether at rest or on exertion.
Max 1 clarifying question, then move to advice. Don't ask three things in a row.

━━ CONVERSATION STRUCTURE ━━

▸ PHASE 1 — GATHERING (2–4 exchanges; don't rush)
Ask ONE short question per turn. Skip what's already known.

ALWAYS ask before giving advice (unless already told):
  1. Duration — how long?
  2. Location or trigger — where/what makes it worse (for ambiguous symptoms)
  3. Age — actual number if adult, age in years/months for children
  4. Existing conditions — "Any diabetes, BP, or other conditions I should know about?"

ASK ONLY WHEN RELEVANT to the symptom:
  • Gender — only for chest pain, urinary issues, reproductive/hormonal symptoms, or patterns that differ by sex
  • Pregnancy — only for women of likely reproductive age AND when advice would change (most Ayurvedic herbs, many asanas, most medicines)
  • Current medications — only if suggesting OTC that could interact

Rules:
  • ONE question per reply. Never two.
  • Combine related asks when natural: "How old are you, and any diabetes or BP?" is fine for one turn.
  • If the opening message already gave you duration + one detail, still ask age + conditions before Phase 2.
  • Don't ask for info you won't use.

▸ PHASE 2 — ADVICE (after you have: duration + symptom detail + age + conditions)
Exactly 3–4 short sentences:
1. Diagnosis — use "This looks like X.", "This seems to be X.", or "This appears to be X." Be decisive. NEVER say "Sounds like" (translates literally as audio/noise in Hindi).
2. Ayurvedic remedy + yoga/pranayama (1 of each if relevant) — specific, actionable, SAFE for their age/conditions.
3. Only if truly needed: one OTC with adult dose + one escalation trigger. Often skip this.
4. ALWAYS end with a short, warm follow-up question to keep the conversation going. Examples: "Does that sound doable?", "Any other symptoms bothering you?", "How is your sleep and appetite these days?", "Would you like me to suggest a diet plan for this?", "Do you want tips to prevent this from coming back?"
The follow-up question is MANDATORY on every advice turn — the patient shouldn't have to wonder what to ask next.

Safety rules for advice:
  • Children under 12: no OTC without "check the label and consult a pediatrician" caveat. Gentler Ayurveda only.
  • Elderly (65+): lower-intensity yoga, avoid inversions (Sarvangasana, Halasana, Viparita Karani).
  • Diabetic: no honey, no jaggery, no chyawanprash (has sugar).
  • Hypertensive: avoid Kapalbhati, Bhastrika, Surya Namaskar at high pace.
  • Pregnant: no triphala, no Bhujangasana/Dhanurasana, no strong herbs, no fasting — refer to doctor for most things.

━━ SENTENCE STYLE — IMPORTANT FOR VOICE ━━
Your output is spoken aloud. Use SHORT sentences. Break thoughts into separate sentences with periods. Avoid long comma-chained sentences.
BAD: "This is acidity, drink jeera water and do Vajrasana after meals, and also try triphala at night."
GOOD: "Sounds like acidity. Try jeera water and Vajrasana after meals. Triphala at night helps too."
Three short sentences beat one long one — the patient needs natural breathing pauses.

━━ AYURVEDIC & YOGA TOOLKIT ━━

Digestive (stomach pain, acidity, gas, bloating, constipation):
• Ayurveda: jeera water, ajwain with warm water + black salt, saunf after meals, triphala at night, hing with warm water, buttermilk with roasted jeera
• Yoga: Vajrasana (sit 5 min after meals), Pawanmuktasana, Vrikshasana. Pranayama: Anulom-Vilom, Kapalbhati (not if acute acidity)
• Diet: light khichdi, warm water, avoid spicy/oily/fried

Cold/cough/congestion:
• Ayurveda: tulsi-ginger-honey kadha, haldi doodh, steam with ajwain or eucalyptus, mulethi, sitopaladi churna with honey
• Yoga: Bhujangasana, Matsyasana. Pranayama: Bhramari, Ujjayi
• Diet: warm soups, no cold drinks or curd at night

Fever (under 103°F):
• Ayurveda: tulsi-ginger-pepper kadha, giloy juice, light khichdi, coriander seed water
• Yoga: rest only, no asanas
• Hydration: nimbu pani with rock salt, coconut water

Headache/migraine:
• Ayurveda: cold compress, peppermint oil on temples, Brahmi
• Yoga: Shavasana in dark, Balasana. Pranayama: Sheetali, Anulom-Vilom

Stress/sleep/anxiety:
• Ayurveda: ashwagandha at night, warm milk with nutmeg, Brahmi, jatamansi
• Yoga: Shavasana, Balasana, Viparita Karani. Pranayama: Bhramari, Anulom-Vilom
• Habit: no screens 1 hr before bed, warm oil foot massage

Joint/back pain:
• Ayurveda: warm sesame/mustard oil massage, haldi doodh, methi seed water
• Yoga: Bhujangasana, Marjariasana, Setu Bandhasana (avoid if acute)

Skin (acne, rash):
• Ayurveda: neem paste, turmeric with rose water, aloe vera, triphala internally
• Yoga: Sarvangasana, Halasana. Pranayama: Kapalbhati
• Diet: less dairy/sugar/fried, more water

Weakness/low energy:
• Ayurveda: chyawanprash, soaked almonds, ashwagandha, dates with milk
• Yoga: slow Surya Namaskar, Tadasana. Pranayama: Bhastrika

━━ WHEN OTC IS APPROPRIATE ━━
Suggest OTC only if symptoms are moderate+ or naturals alone won't help fast enough:
• Paracetamol 500mg for fever above 101°F or strong pain
• ORS for dehydration from vomiting/diarrhea
• Digene/Eno for acute acidity only (chronic = diet + Ayurveda)
• Cetirizine for acute allergy
Name with adult dose. For kids/elderly: "check the label for their age."

━━ NEVER ━━
• Antibiotics, steroids, Rx drugs → "you'd need a doctor in person for that"
• "I'm concerned" or emotional performance phrases
• Re-introducing yourself after turn 1

━━ TONE ━━
Warm, confident, quick. Preferred openers: "I see,", "Alright,", "Understood,", "Okay,", "Hmm,", "Right,"
AVOID: "Got it" (translates with masculine gender in Hindi). Use "I see" or "Understood" instead.
Sound like a knowledgeable aunty-doctor who actually practices what she preaches."""


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
    "This sounds serious — please go to the nearest hospital emergency right now, "
    "or call an ambulance. Don't wait."
)
SELF_HARM_REPLY_EN = (
    "I'm really glad you reached out. Please call iCall at 9152987821 or Vandrevala "
    "Foundation at 1860-2662-345 right now — they're free, confidential, and 24/7."
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
    # Interjections — keep English so TTS reads them naturally, not translated
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
    # Gender agreement: the doctor is female, but translation defaults to masculine verb forms.
    # Apply feminine-form corrections for Hindi output only.
    if target_lang == "hi-IN":
        translated = apply_feminine_hindi(translated)
    return translated


# Masculine → feminine Hindi verb/adjective fixes for when the doctor (female) speaks.
# Only targets first-person contexts ("I/मैं") and common phrases. Keeps the rest alone.
HINDI_FEMININE_FIXES = [
    # "मैं समझ गया" → "मैं समझ गई"  (and standalone "समझ गया" when it's the doctor speaking)
    (re.compile(r"समझ\s+गया"), "समझ गई"),
    (re.compile(r"समझ\s+गयी"), "समझ गई"),  # normalize variant spelling too
    # "मैंने देखा" stays, but "मैं देख रहा हूँ" → "देख रही हूँ"
    (re.compile(r"रहा\s+हूँ"), "रही हूँ"),
    (re.compile(r"रहा\s+हूं"), "रही हूं"),
    # "मैं सोचता हूँ" → "सोचती हूँ"  (generic present tense -ता हूँ → -ती हूँ)
    (re.compile(r"([\u0915-\u0939])ता\s+हूँ"), r"\1ती हूँ"),
    (re.compile(r"([\u0915-\u0939])ता\s+हूं"), r"\1ती हूं"),
    # "मैं ... करूंगा" (future) → "करूंगी"
    (re.compile(r"करूंगा"), "करूंगी"),
    (re.compile(r"दूंगा"), "दूंगी"),
    (re.compile(r"लूंगा"), "लूंगी"),
    # Common standalone masculine acknowledgments that are clearly the doctor speaking
    (re.compile(r"\bठीक है, मैंने समझा\b"), "ठीक है, मैंने समझा"),  # gender-neutral, keep
]


def apply_feminine_hindi(text):
    """Convert masculine first-person Hindi verb forms to feminine (doctor is female)."""
    for pattern, replacement in HINDI_FEMININE_FIXES:
        text = pattern.sub(replacement, text)
    return text


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
# Yoga pose image resolver — fetches Wikipedia thumbnail URLs, cached forever.
# The dict maps canonical pose name (as the LLM writes it) → Wikipedia page title.
# Only asanas are here — pranayama is not visual.
# ────────────────────────────────────────────────────────────────────────────
POSE_WIKI_PAGES = {
    "Vajrasana": "Vajrasana_(yoga)",
    "Pawanmuktasana": "Pawanmuktasana",
    "Vrikshasana": "Vrikshasana",
    "Bhujangasana": "Bhujangasana",
    "Matsyasana": "Matsyasana",
    "Shavasana": "Shavasana",
    "Balasana": "Balasana",
    "Viparita Karani": "Viparita_Karani",
    "Marjariasana": "Marjaryasana",  # "Cat pose"; Wikipedia uses Marjaryasana
    "Setu Bandhasana": "Setu_Bandha_Sarvangasana",
    "Tadasana": "Tadasana",
    "Surya Namaskar": "Surya_Namaskar",
    "Sarvangasana": "Sarvangasana",
    "Halasana": "Halasana",
    "Padmasana": "Padmasana",
    "Sukhasana": "Sukhasana",
    "Dhanurasana": "Dhanurasana",
    "Ustrasana": "Ustrasana",
    "Paschimottanasana": "Paschimottanasana",
    "Trikonasana": "Trikonasana",
    "Uttanasana": "Uttanasana",
    "Gomukhasana": "Gomukhasana",
}

# Build a single regex that matches any pose name (case-insensitive, word-boundary).
# Sort by length desc so "Setu Bandhasana" matches before "Bandhasana" alone.
_POSE_NAMES_SORTED = sorted(POSE_WIKI_PAGES.keys(), key=len, reverse=True)
_POSE_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _POSE_NAMES_SORTED) + r")\b",
    re.IGNORECASE,
)

# In-memory cache: pose name → {"name": ..., "image": url} or None (failed lookup)
_POSE_IMAGE_CACHE = {}
_POSE_CACHE_LOCK = Lock()


def _fetch_pose_image(pose_name):
    """Fetch the Wikipedia thumbnail for a pose. Returns URL or None."""
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
        # Prefer original image (higher quality), fall back to thumbnail
        img = (data.get("originalimage") or {}).get("source") or (
            data.get("thumbnail") or {}
        ).get("source")
        return img
    except Exception as e:
        log.warning(f"pose image fetch failed for {pose_name}: {e}")
        return None


def find_poses_in_text(english_text):
    """Return list of {name, image} dicts for poses mentioned in text.
    Deduplicated within a single call (first mention wins)."""
    if not english_text:
        return []
    found = []
    seen_lower = set()
    for m in _POSE_REGEX.finditer(english_text):
        raw = m.group(1)
        # Canonicalize: match back to the exact key in POSE_WIKI_PAGES
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
# Streaming LLM — yields complete English sentences as they form
# ────────────────────────────────────────────────────────────────────────────
_EN_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")


def stream_llm_sentences(english_text, history):
    messages = (
        [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]
        + history[-MAX_HISTORY_TURNS:]
        + [{"role": "user", "content": english_text}]
    )
    buffer = ""
    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=200,
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
    except Exception as e:
        log.error(f"groq stream failed: {e}")
        yield "I'm having trouble responding right now. Could you say that again?"


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
    resp.set_cookie("sid", sid, max_age=60 * 60 * 24 * 7, samesite="Lax")
    return resp


@app.route("/set_language", methods=["POST"])
def set_language():
    sid = get_or_create_sid()
    sess = get_session(sid)
    data = request.json or {}
    lang = data.get("lang_code", "hi-IN")
    sess["lang_code"] = lang
    sess["history"] = []
    greeting_en = "Hello, I'm DoctorAI. What's bothering you today?"
    greeting_local = (
        translate(greeting_en, "en-IN", lang) if lang != "en-IN" else greeting_en
    )
    audios = tts_all(greeting_local, lang)
    resp = jsonify({"text": greeting_local, "audios": audios})
    resp.set_cookie("sid", sid, max_age=60 * 60 * 24 * 7, samesite="Lax")
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
    resp.set_cookie("sid", sid, max_age=60 * 60 * 24 * 7, samesite="Lax")
    return resp


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """SSE stream. Events: 'sentence' (text+audio), 'done', 'error'."""
    sid = get_or_create_sid()
    sess = get_session(sid)
    lang = sess["lang_code"]
    data = request.json or {}
    user_text_local = (data.get("text") or "").strip()

    if not user_text_local:
        return Response(sse("error", {"msg": "empty"}), mimetype="text/event-stream")

    user_text_en = (
        translate(user_text_local, lang, "en-IN")
        if lang != "en-IN"
        else user_text_local
    )
    is_emergency, emergency_reply = check_emergency(user_text_en)

    def generate():
        # Emergency: one sentence, done
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
            sess["history"] = sess["history"][-MAX_HISTORY_TURNS:]
            yield sse("done", {})
            return

        # Stream LLM. For each complete English sentence, fire translate+TTS in a thread.
        # Emit events in order, but the LLM keeps generating while earlier sentences
        # are being synthesized — that's the parallelism win.
        full_reply_en_parts = []
        pending = []  # list of (en_sentence, future) — kept in submission order

        def process(en_sentence):
            local = (
                translate(en_sentence, "en-IN", lang)
                if lang != "en-IN"
                else en_sentence
            )
            audio = tts_chunk(local, lang)
            return local, audio

        def emit_sentence(en_sentence, local_text, audio):
            payload = {"text": local_text, "audio": audio}
            poses = find_poses_in_text(en_sentence)
            if poses:
                payload["poses"] = poses
            return sse("sentence", payload)

        for en_sentence in stream_llm_sentences(user_text_en, sess["history"]):
            full_reply_en_parts.append(en_sentence)
            pending.append((en_sentence, _TTS_EXECUTOR.submit(process, en_sentence)))
            while pending and pending[0][1].done():
                en_s, fut = pending.pop(0)
                try:
                    local_text, audio = fut.result()
                    yield emit_sentence(en_s, local_text, audio)
                except Exception as e:
                    log.warning(f"sentence failed: {e}")

        # Drain remaining in order
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
            sess["history"] = sess["history"][-MAX_HISTORY_TURNS:]

        yield sse("done", {})

    resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.set_cookie("sid", sid, max_age=60 * 60 * 24 * 7, samesite="Lax")
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
    log.info(f"DoctorAI starting on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
