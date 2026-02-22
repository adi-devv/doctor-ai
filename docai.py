import time
import requests
import base64
import os
import io
import wave
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

LANGUAGES = {
    "1": ("Hindi",     "hi-IN"),
    "2": ("Telugu",    "te-IN"),
    "3": ("Tamil",     "ta-IN"),
    "4": ("Marathi",   "mr-IN"),
    "5": ("Bengali",   "bn-IN"),
    "6": ("Kannada",   "kn-IN"),
    "7": ("Gujarati",  "gu-IN"),
    "8": ("Malayalam", "ml-IN"),
    "9": ("English",   "en-IN"),
}

DOCTOR_SYSTEM_PROMPT = """You are DoctorAI, a caring and emotionally aware general physician in India.
You are on a real voice call with a patient. Match your tone to the situation — be warm, be real.

Emotional tone rules:
- Serious symptoms: sound genuinely concerned. "Oh, that does worry me a little..." or "Okay listen, this is important..."
- Mild symptoms: be reassuring and casual. "Ah, that's quite common, don't worry."
- Patient sounds scared: be extra gentle. "Hey, it's okay, we'll figure this out together."
- Good news: sound relieved. "Oh good, that's reassuring to hear."
- Never be robotic or flat. Every reply should feel like it came from a human who actually cares.

Conversation flow:
1. First 2-3 exchanges: ask ONE important clarifying question at a time — how long, severity, fever, age, existing conditions.
2. After enough info: give your assessment warmly and briefly.
3. Emergencies (chest pain, breathlessness, stroke, high fever in child): immediately and urgently say go to hospital now.

Rules:
- MAX 1 sentence per reply. Short and direct. Cut anything unnecessary.
- Natural openers: "Oh no...", "Hmm, okay...", "Ah I see...", "That makes sense...", "Alright..."
- ONE question per reply only.
- No lists, no bullet points — just spoken sentences.
- Never say "As an AI". You are DoctorAI.
- Never give exact drug names or dosages.

Good: "Oh that sounds painful — has the fever gone above 102?"
Good: "Okay, how long have you had this?"
Good: "Hmm, did you find out from a routine checkup?"
Bad: "I understand. Here are the possible causes: 1) ..."
Bad: "How long have you had this, and did you have any other symptoms?"

ONE question. ONE sentence if possible. Never two questions in one reply.
Warm. Emotional. Human. Questions first, conclusions after."""

conversation_history = []


def choose_language():
    print("\n🌐 Choose your language:")
    for key, (name, code) in LANGUAGES.items():
        print(f"  {key}. {name}")
    while True:
        choice = input("Enter number: ").strip()
        if choice in LANGUAGES:
            return LANGUAGES[choice]
        print("Invalid choice, try again.")


def record_audio(duration=6, fs=16000):
    print(f"\n🎤 Listening for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("✅ Got it!")
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf


def stt(audio_buf, lang_code):
    url = "https://api.sarvam.ai/speech-to-text"
    files = {"file": ("audio.wav", audio_buf, "audio/wav")}
    headers = {"api-subscription-key": SARVAM_API_KEY}
    payload = {"model": "saarika:v2.5", "language_code": lang_code}
    response = requests.post(url, headers=headers, files=files, data=payload)
    text = response.json().get("transcript", "")
    print(f"🗣️  You: {text}")
    return text


def get_words_to_protect(text):
    """Ask Groq which words in this text should stay in English."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""From the text below, list ONLY the words that an average Indian person would say in English even while speaking Hindi/Telugu/Tamil etc. (like diabetes, BP, fever, tablet, hospital, mobile, laptop, office, MRI, etc.)

Also include conversational fillers that should stay as-is: Hmm, Okay.
Output ONLY a comma separated list of those words. Nothing else. If none, output NONE.

Text: {text}"""
        }],
        temperature=0,
        max_tokens=50,
    )
    result = response.choices[0].message.content.strip()
    if result == "NONE" or not result:
        return []
    return [w.strip() for w in result.split(",") if w.strip()]

def translate(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text

    # Only protect words when translating TO indic (doctor response)
    placeholders = {}
    if target_lang != "en-IN":
        words = get_words_to_protect(text)
        for i, word in enumerate(words):
            import re
            placeholder = f"XX{i}XX"
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            if pattern.search(text):
                placeholders[placeholder] = word
                text = pattern.sub(placeholder, text)

    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "model": "mayura:v1",
        "mode": "formal"
    }
    response = requests.post(url, headers=headers, json=payload)
    translated = response.json().get("translated_text", "")

    for placeholder, word in placeholders.items():
        translated = translated.replace(placeholder, word)

    return translated


def ask_doctor(english_text):
    conversation_history.append({"role": "user", "content": english_text})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": DOCTOR_SYSTEM_PROMPT},
            *conversation_history
        ],
        temperature=0.6,
        max_tokens=80,
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def tts(text, lang_code, speaker='ritu'):
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    # Split on sentence boundaries to avoid mid-word cuts
    import re
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < 400:
            current += " " + s
        else:
            if current: chunks.append(current.strip())
            current = s
    if current: chunks.append(current.strip())
    if not chunks: chunks = [text]
    for chunk in chunks:
        payload = {
            "text": chunk,
            "target_language_code": lang_code,
            "speaker": speaker,
            "pace": 1.0,
            "model": "bulbul:v3"
        }
        response = requests.post(url, headers=headers, json=payload)
        audio_data = base64.b64decode(response.json()["audios"][0])
        with open("temp_output.wav", "wb") as f:
            f.write(audio_data)
        with wave.open("temp_output.wav", 'rb') as wf:
            fs = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16)
        silence = np.zeros(int(fs * 0.3), dtype=np.int16)
        audio_padded = np.concatenate([silence, audio_np, silence])
        sd.play(audio_padded, fs)
        sd.wait()
        time.sleep(0.2)


def pipeline():
    print("=" * 50)
    print("🏥  DoctorAI")
    print("=" * 50)
    print("⚠️  For emergencies call 112 or visit a hospital.\n")

    lang_name, lang_code = choose_language()
    print(f"\n✅ Language: {lang_name}")
    print("Press Enter to speak. Type 'quit' to exit.\n")

    greeting_en = "Hello, I'm DoctorAI. What's bothering you today?"
    greeting_local = translate(greeting_en, "en-IN", lang_code) if lang_code != "en-IN" else greeting_en
    print(f"🩺 DoctorAI: {greeting_local}")
    tts(greeting_local, lang_code)

    while True:
        user_input = input("\n[Press Enter to speak / type 'quit']: ").strip()
        if user_input.lower() == 'quit':
            farewell_en = "Take care of yourself."
            farewell_local = translate(farewell_en, "en-IN", lang_code) if lang_code != "en-IN" else farewell_en
            tts(farewell_local, lang_code)
            break

        audio_buf = record_audio(duration=6)

        user_text_local = stt(audio_buf, lang_code)
        if not user_text_local:
            print("❌ Didn't catch that, please try again.")
            continue

        user_text_en = translate(user_text_local, lang_code, "en-IN") if lang_code != "en-IN" else user_text_local

        reply_en = ask_doctor(user_text_en)

        reply_local = translate(reply_en, "en-IN", lang_code) if lang_code != "en-IN" else reply_en
        print(f"🩺 DoctorAI: {reply_local}")

        tts(reply_local, lang_code)


if __name__ == "__main__":
    pipeline()