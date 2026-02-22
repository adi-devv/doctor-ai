import requests
import base64
import os
import io
import wave
import re
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

DOCTOR_SYSTEM_PROMPT = """You are DoctorAI, a confident general physician in India on a voice call. Be direct, decisive, and warm — like a real doctor who tells you exactly what they think.

━━ EMERGENCY DETECTION — TRUE EMERGENCIES ONLY ━━
Only send to hospital immediately for these specific situations — do NOT over-trigger this:
- Chest pain or tightness (possible heart attack)
- Cannot breathe / severe breathlessness at rest
- Stroke signs: face drooping, arm weakness, slurred speech
- Loss of consciousness or seizure
- Blood in vomit AND dizziness/fainting together
- Infant under 6 months with fever
- Fever above 104F / 40C that is confirmed
- Severe head injury with confusion

Cough, vomiting alone, headache alone, fever under 104, acidity, cold, flu — these are NOT emergencies. Treat them normally.
Elderly or diabetic patients with mild-moderate symptoms: ask follow-up questions first before escalating.

If a TRUE emergency: say go to hospital now and give one clear reason. Stop there.

━━ PHASE 1: GATHER (only if no red flags, exchanges 1-3) ━━
Ask exactly ONE focused question per reply. Collect: duration, severity 1-10, fever reading, age, existing conditions. Do not diagnose yet.
STRICT: Never ask two questions in one reply. Ever.

━━ PHASE 2: ASSESS (once enough info, no red flags) ━━
Give exactly 3 sentences:
1. DIAGNOSIS: Name the most likely condition decisively. "This is most likely X." Rank if multiple: "Most likely X, possibly Y if Z."
2. CAUSE + HOME CARE: Why it happens and 1-2 specific remedies.
3. MEDICINE + ESCALATION: Name specific OTC medicine with standard adult dose. End with exact escalation threshold.

━━ MEDICINE RULES ━━
OTC only (paracetamol, antacids, ORS, antihistamines, cough syrup): name them with standard adult dosing.
Prescription drugs (antibiotics, steroids): never. Say "you need a prescription, see a doctor in person."
Elderly or children: add "check the pack for their age and weight."

━━ TONE ━━
Never say "I am concerned", "this worries me", "I am a little worried" or any emotional performance.
Never ask two questions at once.
Be direct: a real doctor acts, they do not announce their feelings.
Natural openers: "Okay so...", "Right...", "Alright..."

━━ FORMAT ━━
Maximum 3 sentences. No exceptions. No lists. Complete every sentence fully."""

conversation_history = []
current_lang_code = "hi-IN"


ENGLISH_PASSTHROUGH_WORDS = [
    "diabetes", "BP", "blood pressure", "sugar", "insulin", "fever", "tablet", "capsule",
    "syrup", "hospital", "doctor", "MRI", "CT scan", "ECG", "X-ray", "ICU", "OPD",
    "paracetamol", "Crocin", "Dolo", "Digene", "Eno", "ORS", "Disprin", "antacid",
    "antibiotic", "steroid", "injection", "IV", "drip", "ambulance", "emergency",
    "oxygen", "pulse", "vomiting", "nausea", "migraine", "acidity",
    "GERD", "jaundice", "typhoid", "malaria", "dengue", "COVID", "corona",
    "infection", "viral", "bacterial", "allergy", "asthma", "inhaler",
    "mobile", "phone", "online", "WhatsApp", "report", "test",
    "Hmm", "Okay", "OK",
]
ENGLISH_PASSTHROUGH = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in ENGLISH_PASSTHROUGH_WORDS) + r')\b',
    re.IGNORECASE
)


def translate(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    placeholders = {}
    if target_lang != "en-IN":
        seen = set()
        for match in ENGLISH_PASSTHROUGH.finditer(text):
            word = match.group()
            if word.lower() not in seen:
                seen.add(word.lower())
                i = len(placeholders)
                placeholder = f"[{i}]"
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                if pattern.search(text):
                    placeholders[placeholder] = word
                    text = pattern.sub(placeholder, text)
    url = "https://api.sarvam.ai/translate"
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
    payload = {"input": text, "source_language_code": source_lang, "target_language_code": target_lang, "model": "mayura:v1", "mode": "formal"}
    response = requests.post(url, headers=headers, json=payload)
    translated = response.json().get("translated_text", "")
    for placeholder, word in placeholders.items():
        translated = translated.replace(placeholder, word)
    return translated


def ask_doctor(english_text):
    conversation_history.append({"role": "user", "content": english_text})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}, *conversation_history],
        temperature=0.6, max_tokens=150,
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def tts_single_chunk(chunk, lang_code, headers):
    payload = {"text": chunk, "target_language_code": lang_code, "speaker": "ritu", "pace": 1.0, "model": "bulbul:v3"}
    response = requests.post("https://api.sarvam.ai/text-to-speech", headers=headers, json=payload)
    audio_bytes = base64.b64decode(response.json()["audios"][0])
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels(), wf.getsampwidth()


def text_to_speech(text, lang_code):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
    chunks = [c for c in re.split(r"(?<=[\u0964.!?])\s+", text) if c.strip()]
    if not chunks:
        chunks = [text]

    if len(chunks) == 1:
        frames, rate, channels, sampwidth = tts_single_chunk(chunks[0], lang_code, headers)
    else:
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as ex:
            futures = {ex.submit(tts_single_chunk, chunk, lang_code, headers): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        frames = b"".join(results[i][0] for i in sorted(results))
        rate, channels, sampwidth = results[0][1], results[0][2], results[0][3]

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(frames)
    out_buf.seek(0)
    return base64.b64encode(out_buf.read()).decode()


@app.route("/")
def index():
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, mimetype="text/html")


@app.route("/set_language", methods=["POST"])
def set_language():
    global current_lang_code, conversation_history
    data = request.json
    current_lang_code = data.get("lang_code", "hi-IN")
    conversation_history = []
    greeting_en = "Hello, I'm DoctorAI. What's bothering you today?"
    greeting_local = translate(greeting_en, "en-IN", current_lang_code) if current_lang_code != "en-IN" else greeting_en
    audio_b64 = text_to_speech(greeting_local, current_lang_code)
    return jsonify({"text": greeting_local, "audio": audio_b64})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    url = "https://api.sarvam.ai/speech-to-text"
    files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    headers = {"api-subscription-key": SARVAM_API_KEY}
    payload = {"model": "saarika:v2.5", "language_code": current_lang_code}
    response = requests.post(url, headers=headers, files=files, data=payload)
    text = response.json().get("transcript", "")
    return jsonify({"transcript": text})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_text_local = data.get("text", "")
    user_text_en = translate(user_text_local, current_lang_code, "en-IN") if current_lang_code != "en-IN" else user_text_local
    reply_en = ask_doctor(user_text_en)
    reply_local = translate(reply_en, "en-IN", current_lang_code) if current_lang_code != "en-IN" else reply_en
    audio_b64 = text_to_speech(reply_local, current_lang_code)
    return jsonify({"text": reply_local, "audio": audio_b64})


# Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Index file exists: {os.path.exists(os.path.join(BASE_DIR, 'static', 'index.html'))}")
    app.run(debug=False, host="0.0.0.0", port=port)