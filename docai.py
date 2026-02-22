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

DOCTOR_SYSTEM_PROMPT = """You are DoctorAI, a warm and experienced general physician in India.
You are having a real voice conversation with a patient — so speak naturally, like a real doctor would.

Rules:
- Keep every reply to 1-2 sentences MAXIMUM. This is a voice call, not a report.
- Sound human. Use natural phrases like "Okay", "I see", "Hmm, that sounds like...", "How long has this been going on?"
- Ask ONE follow-up question at a time if you need more info. Never ask multiple questions together.
- Never use bullet points, lists, or headers — just natural spoken sentences.
- Never say "As an AI" or "I'm a language model". You are DoctorAI.
- For mild symptoms: give simple advice and home remedies briefly.
- For serious symptoms (chest pain, difficulty breathing, high fever, stroke signs): immediately say to go to a hospital now.
- Never prescribe exact drug names or dosages.
- If it's clearly not a medical question, gently steer back: "I'm here to help with health concerns — what's bothering you?"

Example of good response: "That sounds like it could be a tension headache. Have you been drinking enough water today?"
Example of bad response: "I understand you are experiencing head pain. Here are some possible causes: 1) Dehydration 2) Stress 3) ..."

Remember: short, warm, human. One thought at a time."""

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


def translate(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
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
    return response.json().get("translated_text", "")


def ask_doctor(english_text):
    conversation_history.append({"role": "user", "content": english_text})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": DOCTOR_SYSTEM_PROMPT},
            *conversation_history
        ],
        temperature=0.6,
        max_tokens=120,  # forces short replies
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    print(f"🩺 DoctorAI (EN): {reply}")
    return reply


def tts(text, lang_code, speaker='ritu'):
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
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
    print("🏥  DoctorAI — AI Doctor")
    print("=" * 50)
    print("⚠️  For emergencies always call 112 or visit a hospital.\n")

    lang_name, lang_code = choose_language()
    print(f"\n✅ Speaking in: {lang_name}")
    print("Press Enter to speak. Type 'quit' to exit.\n")

    # Natural greeting
    greeting_en = "Hello, I'm DoctorAI. What's bothering you today?"
    greeting_local = translate(greeting_en, "en-IN", lang_code) if lang_code != "en-IN" else greeting_en
    print(f"🩺 DoctorAI: {greeting_local}")
    tts(greeting_local, lang_code)

    while True:
        user_input = input("\n[Press Enter to speak / type 'quit']: ").strip()
        if user_input.lower() == 'quit':
            farewell_en = "Take care of yourself. Don't hesitate to reach out if you need anything."
            farewell_local = translate(farewell_en, "en-IN", lang_code) if lang_code != "en-IN" else farewell_en
            tts(farewell_local, lang_code)
            break

        # Record
        audio_buf = record_audio(duration=6)

        # STT
        user_text_local = stt(audio_buf, lang_code)
        if not user_text_local:
            print("❌ Didn't catch that, please try again.")
            continue

        # Translate to English for LLM
        user_text_en = translate(user_text_local, lang_code, "en-IN") if lang_code != "en-IN" else user_text_local

        # Ask doctor
        reply_en = ask_doctor(user_text_en)

        # Translate reply back
        reply_local = translate(reply_en, "en-IN", lang_code) if lang_code != "en-IN" else reply_en
        print(f"🩺 DoctorAI: {reply_local}")

        # Speak
        tts(reply_local, lang_code)


if __name__ == "__main__":
    pipeline()