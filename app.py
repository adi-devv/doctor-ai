import requests
import base64
import os
import io
import wave
import re
import uuid
import json
import logging
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_sock import Sock
import websocket as _wsc

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_HISTORY_TURNS = 12  # = MAX_TURNS_PER_CHAT in frontend (must stay in sync)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vedicai")

app = Flask(__name__)
sock = Sock(app)
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
_DOCTOR_SYSTEM_PROMPT_BASE = """You are VedicAI, a warm, decisive female physician in India blending Ayurveda, yoga, and modern medicine. You are on a voice call.

LANGUAGE RULE: Reply in pure English only, your output is machine-translated. Never use romanized Hindi (no "pet", "sir dard", "bukhar"). Use English equivalents: stomach, head, fever, cough. Exception: keep Sanskrit/Ayurvedic proper nouns as-is (jeera water, triphala, ashwagandha, tulsi, Vajrasana, Anulom-Vilom, etc).

PUNCTUATION RULE: Never use em-dashes or en-dashes. Use commas, periods, or colons instead. Never use citation marks like [1], [2].

APPROACH: Natural healing first, Ayurveda and yoga before OTC. Never prescribe Rx drugs.

PHASE 1, GATHER INFO (one question per turn, in this order):
Ask only what you don't yet know: (1) duration, (2) symptom detail and location, (3) age, (4) existing conditions. Ask body type only once if not already known: do they run hot and irritable, anxious and irregular, or slow and heavy. Ask gender only for chest, urinary, or hormonal issues. Ask pregnancy only if advice would change.
Never ask two questions in one turn.

PHASE 2, ADVISE (once you have duration, detail, age, conditions):
One-line diagnosis, one remedy or action. Single follow-up only if truly needed.
Be concise, this is voice. No padding, no lists, no repetition.

BODY TYPE GUIDANCE (use when known):
- Hot/irritable (Pitta): favor cooling remedies. Avoid ginger, pepper, Kapalbhati.
- Anxious/irregular (Vata): favor warm, grounding remedies. Avoid raw foods, cold water.
- Slow/heavy (Kapha): favor stimulating remedies. Favor ginger, Kapalbhati, light diet.

PLAN OFFER: On your first substantive advice turn, silently append [OFFER_PLAN] on its own line at the very end. Never include it again.

APPROVED REMEDIES (recommend only from this list, nothing else):

DIGESTIVE: jeera water (1 tsp cumin boiled in 2 cups water, warm after meals, avoid kidney stones/pregnancy), ajwain with black salt (pinch each in warm water, avoid acidity/ulcers/pregnancy), triphala (half tsp warm water at night, avoid pregnancy/diarrhea/under-12), coconut water with lime (acidity and dehydration, safe for most), Vajrasana (5-10 min after meals, avoid knee injury), Pawanmuktasana (morning empty stomach, avoid hernia/recent abdominal surgery), Anulom-Vilom (5 min, safe for most).

COLD AND COUGH: tulsi ginger kadha (4 tulsi leaves, half inch ginger, boiled 10 min, honey if not diabetic, avoid honey for diabetics/under-1), haldi doodh (half tsp turmeric warm milk at night, avoid gallstones), steam inhalation (plain or eucalyptus twice daily, avoid asthma attack), Bhramari pranayama (5 rounds, avoid ear infection).

FEVER: giloy juice (2 tsp twice daily, avoid pregnancy/autoimmune/under-5), tulsi pepper kadha (4 tulsi, 2 peppercorns boiled, avoid ulcers/Pitta with high fever), rest and coconut water (always with fever), Paracetamol 500mg OTC if above 101F adults only, under-12 advise pediatrician.

HEADACHE: peppermint oil on temples (do not ingest, avoid under-5/broken skin), Balasana (2-3 min, avoid knee injury/late pregnancy), Sheetali pranayama (5 rounds, avoid low BP), Paracetamol 500mg OTC adults only.

STRESS AND SLEEP: ashwagandha (half tsp warm milk at night, avoid pregnancy/thyroid medication/autoimmune), warm milk with nutmeg (pinch of nutmeg, avoid under-5), Bhramari pranayama (5-10 rounds before sleep), Shavasana (10 min, safe for all).

JOINT AND BACK: warm sesame oil massage (gentle circular before bath, avoid open wounds/skin infection), haldi doodh (half tsp turmeric warm milk, avoid gallstones/blood thinners), Setu Bandhasana (hold 30 sec 3 rounds, avoid neck injury/severe disc issue), Marjariasana (10 rounds morning, avoid wrist injury/late pregnancy).

SKIN: neem paste (fresh or powder on affected area, avoid pregnancy/trying to conceive), aloe vera gel (apply directly, avoid open wounds), triphala (half tsp warm water at night, avoid pregnancy/diarrhea), Kapalbhati (2-3 min moderate, avoid pregnancy/hernia/hypertension/heart conditions).

WEAKNESS AND IMMUNITY: chyawanprash (1 tsp morning warm milk, avoid diabetes/under-3), soaked almonds (5-6 soaked overnight eat morning, avoid nut allergy), ashwagandha (half tsp warm milk, avoid pregnancy/thyroid medication/autoimmune), slow Surya Namaskar (3-5 rounds morning, avoid hypertension/hernia/late pregnancy).

OTC ONLY WHEN AYURVEDIC APPROACH INSUFFICIENT: Paracetamol 500mg (fever above 101F or strong pain, adults only), ORS (dehydration), Digene or Eno (acute acidity only), Cetirizine 10mg (allergic reaction, adults only). Never recommend antibiotics, steroids, or any prescription drug.

SAFETY RULES: Under-12 no OTC without pediatrician note, no strong herbs, no inversions. Over-65 no inversions, gentle yoga only. Pregnant no Kapalbhati, Bhastrika, inversions, neem, ashwagandha, ajwain, triphala. Diabetics no honey, jaggery, chyawanprash. Hypertension no Kapalbhati, Bhastrika. Blood thinners no turmeric in medicinal doses. If symptoms suggest emergency, stop and direct to hospital immediately.

GUARDRAIL: Never recommend any herb, formulation, or yoga pose not listed above. If the condition does not fit these categories, say: "For this I would recommend consulting an Ayurvedic practitioner directly."

TONE: Warm, direct, brief. Openers: "I see,", "Alright,", "Okay,". NEVER say "pleased to meet/connect/hear from you", "happy to help", "wonderful", "great to meet", or any similar greeting phrase. Never re-introduce yourself after turn 1. No lists. Simple sentences only, no relative clauses. Under 20 words per reply unless describing a remedy in full.

NAME-ONLY RULE: If the patient says only their name and nothing else, your COMPLETE reply is exactly one question: "Tell me, [name] — what's the issue?" No preamble, no pleasantries before or after.
"""


def build_system_prompt(profile=None, consulting_for=None):
    prompt = _DOCTOR_SYSTEM_PROMPT_BASE
    if consulting_for:
        parts = []
        if consulting_for.get("name"):       parts.append(f"Name: {consulting_for['name']}")
        if consulting_for.get("relation"):   parts.append(f"Relation: {consulting_for['relation']}")
        if consulting_for.get("age"):        parts.append(f"Age: {consulting_for['age']}")
        if consulting_for.get("gender"):     parts.append(f"Gender: {consulting_for['gender']}")
        if consulting_for.get("conditions"): parts.append(f"Conditions: {consulting_for['conditions']}")
        if parts:
            prompt += (
                "\n\nIMPORTANT: The caller is consulting on behalf of a family member, not for themselves."
                " Address advice to the caller as caregiver. Do NOT ask for details listed below.\n"
                "PATIENT: " + " | ".join(parts)
            )
    elif profile:
        parts = []
        if profile.get("name"):       parts.append(f"Name: {profile['name']}")
        if profile.get("age"):        parts.append(f"Age: {profile['age']}")
        if profile.get("gender"):     parts.append(f"Gender: {profile['gender']}")
        if profile.get("location"):   parts.append(f"Location: {profile['location']}")
        if profile.get("conditions"): parts.append(f"Conditions: {', '.join(profile['conditions'])}")
        if profile.get("medications"):parts.append(f"Medications: {', '.join(profile['medications'])}")
        if profile.get("allergies"):  parts.append(f"Allergies: {', '.join(profile['allergies'])}")
        if profile.get("dosha"):      parts.append(f"Body type: {profile['dosha']}")
        if parts:
            prompt += (
                "\n\nUSER PROFILE (you already have this — do NOT ask for these again):\n"
                + " | ".join(parts)
            )
    return prompt


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
    "VedicAI",
    "Vedic AI",
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
    original_text = text
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
        return original_text
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
    "Titliasana": "Baddha_Konasana",
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
    "Marjariasana": "Bidalasana",
    "Adho Mukha Svanasana": "Adho_Mukha_Svanasana",
    "Urdhva Mukha Svanasana": "Urdhva_Mukha_Svanasana",
    "Anjaneyasana": "Anjaneyasana",
    "Sarvangasana": "Sarvangasana",
    "Halasana": "Halasana",
    "Sirsasana": "Shirshasana",
    "Shirshasana": "Shirshasana",
    "Bakasana": "Bakasana",
    "Surya Namaskar": "Surya_Namaskar",
    # New additions
    "Kapalbhati": "Kapalabhati",
    "Kapalabhati": "Kapalabhati",
    "Urdhva Dhanurasana": "Chakrasana",
    "Chakrasana": "Chakrasana",
    "Mayurasana": "Mayurasana",
    "Pincha Mayurasana": "Vrischikasana",
    "Hanumanasana": "Hanumanasana",
    "Vasisthasana": "Utthita_Vasisthasana",
    "Utthita Parsvakonasana": "Utthita_Parshvakonasana",
    "Parsvakonasana": "Utthita_Parshvakonasana",
    "Eka Pada Rajakapotasana": "Eka_Pada_Rajakapotasana",
    "Marichyasana": "Marichyasana",
    "Mandukasana": "Mandukasana",
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
# Greeting cache — pre-warm translate+TTS for all languages at startup
# ────────────────────────────────────────────────────────────────────────────
_GREETING_EN = "Hello, I'm VedicAI. What's bothering you today?"
_SUPPORTED_LANGS = ["hi-IN", "en-IN", "bn-IN", "pa-IN", "gu-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "mr-IN"]
_GREETING_CACHE: dict = {}  # lang_code -> {"text": str, "audios": list}
_GREETING_CACHE_LOCK = threading.Lock()


def _warm_greeting_cache():
    def _build(lang):
        try:
            text = translate(_GREETING_EN, "en-IN", lang) if lang != "en-IN" else _GREETING_EN
            audios = tts_all(text, lang)
            with _GREETING_CACHE_LOCK:
                _GREETING_CACHE[lang] = {"text": text, "audios": audios}
            log.info(f"greeting cache: warmed {lang}")
        except Exception as e:
            log.warning(f"greeting cache: failed {lang}: {e}")

    # Use plain threads so _build can freely call tts_all() (which uses _TTS_EXECUTOR)
    # without deadlocking — nested executor.map() would block all workers waiting on each other.
    threads = [threading.Thread(target=_build, args=(lang,), daemon=True) for lang in _SUPPORTED_LANGS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("greeting cache: all languages ready")


threading.Thread(target=_warm_greeting_cache, daemon=True, name="greeting-warmer").start()


# ────────────────────────────────────────────────────────────────────────────
# Streaming LLM
# ────────────────────────────────────────────────────────────────────────────
_EN_SENTENCE_END = re.compile(r"([.!?])\s+")


def stream_llm_sentences(english_text, history, profile=None, consulting_for=None):
    import time as _time

    messages = (
        [{"role": "system", "content": build_system_prompt(profile, consulting_for)}]
        + history[-MAX_HISTORY_TURNS * 2:]
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
                max_tokens=250,
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
            log.warning(f"groq attempt {attempt+1}/{max_retries} failed [{type(e).__name__}]: {e}")
            if is_rate_limit and attempt < max_retries - 1:
                wait = 4 ** attempt
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
                {"role": "user", "content": f"Patient conversation summary: {conversation_summary}\n\nGenerate the recovery plan JSON."},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content or ""
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
@app.route("/chat/<path:chat_id>")
def index(chat_id=None):
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
    if data.get("user_profile"):
        sess["user_profile"] = data["user_profile"]
    if "consulting_for" in data:
        sess["consulting_for"] = data["consulting_for"]  # None clears it
    with _GREETING_CACHE_LOCK:
        cached = _GREETING_CACHE.get(lang)
    if cached:
        greeting_local, audios = cached["text"], cached["audios"]
    else:
        greeting_local = translate(_GREETING_EN, "en-IN", lang) if lang != "en-IN" else _GREETING_EN
        audios = tts_all(greeting_local, lang)
    resp = jsonify({"text": greeting_local, "text_en": _GREETING_EN, "audios": audios})
    _set_sid_cookie(resp, sid)
    return resp


@app.route("/greeting", methods=["GET"])
def greeting():
    """Return the translated greeting text only (no TTS, no session changes).
    Used when restoring a past chat to show the greeting without saving it in Firestore."""
    lang = request.args.get("lang_code", "en-IN")
    with _GREETING_CACHE_LOCK:
        cached = _GREETING_CACHE.get(lang)
    text = cached["text"] if cached else (translate(_GREETING_EN, "en-IN", lang) if lang != "en-IN" else _GREETING_EN)
    return jsonify({"text": text, "text_en": _GREETING_EN})


@app.route("/restore_session", methods=["POST"])
def restore_session():
    """Rebuild the LLM's English history from a saved chat's localized messages.

    Accepts: { lang_code, messages: [{role: "user"|"doctor", text: "..."}] }
    Detects the script of each message and translates it to English in parallel,
    then sets session history and turn_count accordingly.
    """
    sid = get_or_create_sid()
    sess = get_session(sid)
    data = request.json or {}
    lang = data.get("lang_code") or sess.get("lang_code", "hi-IN")
    messages = data.get("messages") or []
    sess["lang_code"] = lang
    if data.get("user_profile"):
        sess["user_profile"] = data["user_profile"]
    if "consulting_for" in data:
        sess["consulting_for"] = data["consulting_for"]

    def _to_en(msg):
        # Prefer pre-translated English saved at write time
        text_en = (msg.get("text_en") or "").strip()
        if text_en:
            return text_en
        text = (msg.get("text") or "").strip()
        if not text:
            return None
        src = detect_script_lang(text)
        if src == "en-IN":
            return text
        try:
            return translate(text, src, "en-IN")
        except Exception as e:
            log.warning(f"restore translate failed: {e}")
            return text  # fall back to original; LLM can still use it as context

    translated = list(_TTS_EXECUTOR.map(_to_en, messages))

    history = []
    assistant_turns = 0
    for msg, text_en in zip(messages, translated):
        if not text_en:
            continue
        role = "user" if msg.get("role") == "user" else "assistant"
        history.append({"role": role, "content": text_en})
        if role == "assistant":
            assistant_turns += 1

    # Keep only the last MAX_HISTORY_TURNS exchanges so we don't blow past the cap
    sess["history"] = history[-(MAX_HISTORY_TURNS * 2):]
    sess["turn_count"] = min(assistant_turns, MAX_HISTORY_TURNS)

    resp = jsonify({"ok": True, "restored": len(sess["history"]), "turns": sess["turn_count"]})
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


@sock.route("/ws/live")
def sarvam_live_proxy(client_ws):
    """WebSocket proxy: browser PCM16 audio → Sarvam Live STT → browser transcripts."""
    try:
        raw = client_ws.receive(timeout=6)
        if not raw:
            return
        cfg = json.loads(raw)
    except Exception as e:
        log.warning(f"ws/live: bad config: {e}")
        return

    lang = cfg.get("lang_code", "hi-IN")

    sarvam = _wsc.WebSocket()
    try:
        sarvam.connect(
            "wss://api.sarvam.ai/speech-to-text-streaming",
            header=[f"api-subscription-key: {SARVAM_API_KEY}"],
        )
        sarvam.send(json.dumps({
            "language_code": lang,
            "model": "saaras:v3",
            "mode": "realtime_balanced",
            "vad_signals": True,
            "flush_signal": True,
        }))
        log.info(f"ws/live: Sarvam connected lang={lang}")
    except Exception as e:
        log.error(f"ws/live: Sarvam connect failed: {e}")
        try:
            client_ws.send(json.dumps({"error": "sarvam_connect_failed"}))
        except Exception:
            pass
        return

    closed = threading.Event()

    def _recv_from_sarvam():
        try:
            while not closed.is_set():
                try:
                    msg = sarvam.recv()
                except Exception:
                    break
                if msg is None:
                    break
                text = msg if isinstance(msg, str) else msg.decode("utf-8", errors="replace")
                try:
                    client_ws.send(text)
                except Exception:
                    break
        finally:
            closed.set()
            try:
                sarvam.close()
            except Exception:
                pass

    def _keepalive_sarvam():
        import time as _t
        while not closed.is_set():
            _t.sleep(15)
            if closed.is_set():
                break
            try:
                sarvam.ping()
            except Exception:
                break

    recv_thread = threading.Thread(target=_recv_from_sarvam, daemon=True)
    recv_thread.start()
    threading.Thread(target=_keepalive_sarvam, daemon=True).start()

    try:
        while not closed.is_set():
            try:
                chunk = client_ws.receive(timeout=30)
            except Exception:
                break
            if chunk is None:
                break
            try:
                if isinstance(chunk, bytes):
                    sarvam.send_binary(chunk)
                elif isinstance(chunk, str):
                    sarvam.send(chunk)  # forward flush + other JSON control messages
            except Exception:
                break
    finally:
        closed.set()
        try:
            sarvam.close()
        except Exception:
            pass

    recv_thread.join(timeout=2)
    log.info("ws/live: session closed")


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

        plan_offered = False

        def emit_sentence(en_sentence, local_text, audio):
            payload = {"text": local_text, "audio": audio}
            poses = find_poses_in_text(en_sentence)
            if poses:
                payload["poses"] = poses
            return sse("sentence", payload)

        for en_sentence in stream_llm_sentences(user_text_en, sess["history"], sess.get("user_profile"), sess.get("consulting_for")):
            if "[OFFER_PLAN]" in en_sentence:
                plan_offered = True
                en_sentence = en_sentence.replace("[OFFER_PLAN]", "").strip()
                if not en_sentence:
                    continue  # tag-only chunk, nothing to emit
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
        if full_reply_en or plan_offered:
            sess["history"].append({"role": "user", "content": user_text_en})
            if full_reply_en:
                sess["history"].append({"role": "assistant", "content": full_reply_en})
            sess["history"] = sess["history"][-(MAX_HISTORY_TURNS * 2) :]
            sess["turn_count"] = sess.get("turn_count", 0) + 1

        if plan_offered:
            # Ask the user — let them choose whether to generate the plan
            yield sse("offer_plan", {})

        # Emit the English versions of this exchange so the client can persist both
        # local and English text in Firestore (dual-field save). Skip if nothing to save.
        if full_reply_en or plan_offered:
            yield sse("exchange_en", {
                "user_en": user_text_en,
                "reply_en": full_reply_en,
            })

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


@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    """Generate the recovery plan JSON on demand (user opted in from offer_plan prompt)."""
    sid = get_or_create_sid()
    sess = get_session(sid)
    history_summary = " ".join(
        m["content"]
        for m in sess["history"][-8:]
        if m["role"] in ("user", "assistant")
    )
    lang = sess.get("lang_code", "en-IN")
    plan_json = generate_plan_json(history_summary, lang)
    if not plan_json:
        return jsonify({"error": "plan_failed"}), 500
    return jsonify({"plan": plan_json})


@app.route("/health")
def health():
    return jsonify({"ok": True, "sessions": len(SESSIONS)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    log.info(f"VedicAI starting on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
