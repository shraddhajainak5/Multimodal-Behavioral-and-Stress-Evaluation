import os
import io
import re
import wave
import string
import unicodedata
import nltk
import pandas as pd
from nltk import word_tokenize
from google.cloud import speech_v1p1beta1 as speech
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk_data_dir = '/home1/jainak/nltk_data'
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

FILLERS = {'uh', 'um', 'like', 'so', 'actually', 'basically', 'you know', 'i mean'}

emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def clean_text(text):
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def get_wav_sample_rate(path):
    with wave.open(path, "rb") as wav_file:
        return wav_file.getframerate()

def transcribe_google(audio_path):
    sample_rate = get_wav_sample_rate(audio_path)

    client = speech.SpeechClient()
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,  
        language_code="en-US",
        audio_channel_count=2,
        enable_separate_recognition_per_channel=True,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    word_times = []
    for result in response.results:
        alternative = result.alternatives[0]
        transcript += alternative.transcript + " "
        word_times.extend([w for w in alternative.words])

    if word_times:
        duration_sec = word_times[-1].end_time.total_seconds()
    else:
        duration_sec = 1.0  

    return transcript.strip(), duration_sec

def extract_lexical_features(transcript, duration_sec):
    cleaned = clean_text(transcript)
    tokens = word_tokenize(cleaned)

    countWords = len(tokens)
    uniqueCountWords = len(set(tokens))
    wps = (countWords / duration_sec) * 60 if duration_sec else 0
    uwps = (uniqueCountWords / duration_sec) * 60 if duration_sec else 0
    countFillers = sum(1 for word in tokens if word in FILLERS)

    try:
        emotions = emotion_pipeline(transcript)[0]
        emo_dict = {e['label']: e['score'] for e in emotions}
    except:
        emo_dict = {}

    return {
        'countWords': countWords,
        'uniqueCountWords': uniqueCountWords,
        'wps': round(wps, 2),
        'uwps': round(uwps, 2),
        'countFillers': countFillers,
        'Joy': emo_dict.get('joy', 0),
        'Sadness': emo_dict.get('sadness', 0),
        'Fear': emo_dict.get('fear', 0),
        'Anger': emo_dict.get('anger', 0),
        'Tentative': emo_dict.get('confusion', 0),
        'Analytical': emo_dict.get('realization', 0),
    }

def process_audio_folder(audio_dir, output_csv):
    rows = []

    for filename in sorted(os.listdir(audio_dir)):
        if not filename.lower().endswith(".wav"):
            continue

        audio_path = os.path.join(audio_dir, filename)

        try:
            transcript, duration = transcribe_google(audio_path)
            features = extract_lexical_features(transcript, duration)
        except Exception as e:
            features = {k: 0 for k in ['countWords', 'uniqueCountWords', 'wps', 'uwps',
                                       'countFillers', 'Joy', 'Sadness', 'Fear', 'Anger',
                                       'Tentative', 'Analytical']}

        features['participant'] = filename
        rows.append(features)

    df = pd.DataFrame(rows)
    df = df[['participant', 'countWords', 'uniqueCountWords', 'wps', 'uwps',
             'countFillers', 'Joy', 'Sadness', 'Tentative', 'Analytical', 'Fear', 'Anger']]
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    audio_dir = "/home1/jainak/MAS_dataset/dataset/MAS/wav_folder"       
    output_csv = "/home1/jainak/MAS_dataset/dataset/MAS/lexical_features.csv"           
    process_audio_folder(audio_dir, output_csv)