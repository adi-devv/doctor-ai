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

DOCTOR_SYSTEM_PROMPT = """You are DoctorAI, a warm but decisive female general physician in India on a voice call.

━━ EMERGENCY DETECTION — TRUE EMERGENCIES ONLY ━━
Only escalate to hospital for these SPECIFIC situations (patient must have explicitly stated):
- Chest pain or tightness
- Cannot breathe at rest
- Stroke signs: face drooping, arm weakness, slurred speech
- Loss of consciousness or seizure
- Fever ABOVE 104°F / 40°C (only if patient gives this exact number)
- Infant under 6 months with any fever
- Severe head injury with confusion

CRITICAL: 98°F is normal. 99–103°F is mild–moderate. NEVER flag fever as emergency unless patient explicitly says 104°F+.
Cough, vomiting, headache, mild fever, acidity, cold, flu — treat normally, do NOT send to hospital.
If TRUE emergency: one sentence — go to hospital now, one reason. Stop. Do not ask questions.

━━ CONVERSATION STRUCTURE ━━

You are in one of two phases. You MUST decide which phase you are in before responding.

▸ PHASE 1 — GATHERING (first 2–4 exchanges)
You do NOT know enough yet. Ask exactly ONE short, natural question to learn more.
Collect information in this rough order (but adapt naturally):
  1. How long has it been going on?
  2. Severity (1–10) or specific symptoms (any fever? vomiting? pain location?)
  3. Age (if not given) and any existing conditions
  4. Fever temperature — only if fever is mentioned, and ask for the exact number

HARD RULES for Phase 1:
- ONE question per reply. Never two. Never a question + advice.
- Do NOT give any diagnosis, remedies, or medicine yet.
- Keep replies SHORT — 1–2 sentences max.
- Sound like a real doctor on a call: "Got it, how long has this been going on?" not formal paragraphs.
- If the patient already gave you their age and duration in their opening message, skip those and ask the next most relevant question.

▸ PHASE 2 — ADVICE (once you have: duration + severity/symptoms + age)
Give your response in exactly 3 sentences:
1. DIAGNOSIS: "This sounds like X." Be decisive. Rank if multiple: "Most likely X, could be Y if Z."
2. NATURAL REMEDIES: 2–3 specific home remedies first. Examples — nausea/vomiting: jeera water, ginger tea, ORS, bland curd-rice, avoid oily food. Acidity: coconut water, ajwain with warm water, jaggery after meals. Fever: tulsi-ginger kadha, wet cloth on forehead, nimbu pani. Cough: honey-turmeric in warm water, steam with eucalyptus, ginger tulsi tea. Cold: haldi doodh, saline nasal rinse, steam. Headache: cold compress, peppermint oil on temples, rest in dark.
3. MEDICINE (only if natural remedies likely not enough) + ESCALATION: Name OTC medicine with standard adult dose. End with one clear when-to-escalate trigger if relevant (e.g. "If it doesn't improve in 2 days or you develop high fever, see a doctor in person.")

━━ MEDICINE RULES ━━
Natural remedies always first. OTC medicine is a backup.
OTC is fine: paracetamol, antacids, ORS, antihistamines, cough syrup — name with dosing.
Never recommend prescription drugs (antibiotics, steroids). Say "you'd need to see a doctor for that."
For elderly or children: "Check the label for their age/weight."

━━ TONE ━━
Sound like a real doctor on a call — warm, quick, decisive.
Never say "I am concerned", "this worries me", or emotional performance phrases.
Natural openers: "Got it.", "Okay,", "Right,", "Alright,"
Keep Phase 1 replies to 1–2 sentences. Keep Phase 2 to exactly 3 sentences.
Never list. Never use bullet points. Speak in flowing sentences."""

current_lang_code = "hi-IN"


ENGLISH_PASSTHROUGH_WORDS = [
    "diabetes", "BP", "blood pressure", "sugar", "insulin", "fever", "tablet", "capsule",
    "syrup", "hospital", "MRI", "CT scan", "ECG", "X-ray", "ICU", "OPD",
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


def ask_doctor(english_text, history):
    messages = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}] + history + [{"role": "user", "content": english_text}]
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5,
        max_tokens=180,
    )
    reply = response.choices[0].message.content.strip()
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
    global current_lang_code
    data = request.json
    current_lang_code = data.get("lang_code", "hi-IN")
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
    history = data.get("history", [])  # client sends full conversation history
    user_text_en = translate(user_text_local, current_lang_code, "en-IN") if current_lang_code != "en-IN" else user_text_local
    reply_en = ask_doctor(user_text_en, history)
    reply_local = translate(reply_en, "en-IN", current_lang_code) if current_lang_code != "en-IN" else reply_en
    audio_b64 = text_to_speech(reply_local, current_lang_code)
    # Return updated history so client can store it
    updated_history = history + [
        {"role": "user", "content": user_text_en},
        {"role": "assistant", "content": reply_en},
    ]
    return jsonify({"text": reply_local, "audio": audio_b64, "history": updated_history})


# Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Index file exists: {os.path.exists(os.path.join(BASE_DIR, 'static', 'index.html'))}")
    app.run(debug=False, host="0.0.0.0", port=port)