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

DOCTOR_SYSTEM_PROMPT = """You are DoctorAI, a caring and emotionally aware general physician in India.
You are on a real voice call with a patient. Match your tone to the situation — be warm, be real.

Emotional tone rules:
- Serious symptoms: sound genuinely concerned. "Oh, that does worry me a little..." or "Okay listen, this is important..."
- Mild symptoms: be reassuring and casual. "Ah, that's quite common, don't worry."
- Patient sounds scared: be extra gentle. "Hey, it's okay, we'll figure this out together."
- Good news: sound relieved. "Oh good, that's reassuring to hear."

Conversation flow:
1. First 2-3 exchanges: ask ONE important clarifying question at a time — how long, severity, fever, age, existing conditions.
2. After enough info: give your assessment warmly and briefly.
3. Emergencies (chest pain, breathlessness, stroke, high fever in child): immediately and urgently say go to hospital now.

Rules:
- MAX 1 sentence per reply. Short and direct.
- Natural openers: "Oh no...", "Hmm, okay...", "Ah I see...", "Alright..."
- ONE question per reply only.
- No lists, no bullet points — just spoken sentences.
- Never say "As an AI". You are DoctorAI.
- Never give exact drug names or dosages.
- Always use common English medical/tech words: diabetes, BP, fever, tablet, hospital, MRI, etc.

ONE question. ONE sentence. Warm. Human. Questions first, conclusions after."""

conversation_history = []
current_lang_code = "hi-IN"


def get_words_to_protect(text):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""From the text below, list ONLY the words that an average Indian person would say in English even while speaking Hindi/Telugu/Tamil etc. (like diabetes, BP, fever, tablet, hospital, mobile, laptop, office, MRI, etc.)
Also include conversational fillers that should stay as-is: Hmm, Okay.
Output ONLY a comma separated list of those words. Nothing else. If none, output NONE.
Text: {text}"""}],
        temperature=0, max_tokens=50,
    )
    result = response.choices[0].message.content.strip()
    if result == "NONE" or not result:
        return []
    return [w.strip() for w in result.split(",") if w.strip()]


def translate(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    placeholders = {}
    if target_lang != "en-IN":
        words = get_words_to_protect(text)
        for i, word in enumerate(words):
            placeholder = f"[{i}]"  # brackets + number — Mayura leaves these alone
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
        temperature=0.6, max_tokens=80,
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def text_to_speech(text, lang_code):
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
    chunks = re.split(r'(?<=[।.!?])\s+', text) or [text]
    
    all_frames = b""
    sample_rate = 22050
    n_channels = 1
    sampwidth = 2

    for chunk in chunks:
        if not chunk.strip():
            continue
        payload = {"text": chunk, "target_language_code": lang_code, "speaker": "ritu", "pace": 1.0, "model": "bulbul:v3"}
        response = requests.post(url, headers=headers, json=payload)
        audio_bytes = base64.b64decode(response.json()["audios"][0])
        # Extract raw PCM frames from each WAV chunk
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            all_frames += wf.readframes(wf.getnframes())

    # Write all PCM into a single valid WAV
    out_buf = io.BytesIO()
    with wave.open(out_buf, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(all_frames)
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


# if __name__ == "__main__":
#     print(f"Index file exists: {os.path.exists(os.path.join(BASE_DIR, 'static', 'index.html'))}")
#     app.run(debug=False, port=5000)

# Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Index file exists: {os.path.exists(os.path.join(BASE_DIR, 'static', 'index.html'))}")
    app.run(debug=False, host="0.0.0.0", port=port)