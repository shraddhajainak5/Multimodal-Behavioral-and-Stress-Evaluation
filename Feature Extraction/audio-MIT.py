import os
import csv
import numpy as np
import librosa

audio_folder = "/project/msoleyma_1026/MIT_INTERVIEW_DATASET/Audio"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

output_csv = "extracted_features.csv"
feature_data = []

header = [
    "File", "MFCCs Shape", "Chroma Shape", "Pitch Mean", "Energy Mean",
    "Spectral Centroid Mean", "Spectral Bandwidth Mean", "Spectral Contrast Mean",
    "Spectral Rolloff Mean", "Zero Crossing Rate Mean", "Tempo (BPM)"
]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)
    y, sr = librosa.load(file_path, sr=16000)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    pitch, voicing, confidence = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    energy = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    features = [
        file, 
        str(mfccs.shape), 
        str(chroma.shape), 
        np.nanmean(pitch), 
        np.mean(energy), 
        np.mean(spectral_centroid), 
        np.mean(spectral_bandwidth), 
        np.mean(spectral_contrast), 
        np.mean(spectral_rolloff), 
        np.mean(zcr), 
        tempo
    ]
    feature_data.append(features)

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(feature_data)

print(f"Processing completed for all audio files. Features saved to {output_csv}")
