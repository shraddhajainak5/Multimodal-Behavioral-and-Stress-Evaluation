import librosa
import numpy as np
import os
import pandas as pd
import parselmouth

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        energy = np.sum(y ** 2) / len(y)
        rms = librosa.feature.rms(y=y)[0]
        power = np.mean(rms**2)
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        pitches = pitch[pitch > 0]
        min_pitch = np.min(pitches) if len(pitches) > 0 else np.nan
        max_pitch = np.max(pitches) if len(pitches) > 0 else np.nan
        mean_pitch = np.mean(pitches) if len(pitches) > 0 else np.nan
        pitch_sd = np.std(pitches) if len(pitches) > 0 else np.nan
        pitch_abs = np.mean(np.abs(pitches)) if len(pitches) > 0 else np.nan
        pitch_quant = np.percentile(pitches, 25) if len(pitches) > 0 else np.nan
        pitchUvsVRatio = np.count_nonzero(pitches == 0) / len(y) if len(y) > 0 else np.nan
        diffPitchMaxMin = max_pitch - min_pitch if min_pitch and max_pitch else np.nan
        diffPitchMaxMean = max_pitch - mean_pitch if max_pitch and mean_pitch else np.nan
        diffPitchMaxMode = max_pitch - pitch_quant if max_pitch and pitch_quant else np.nan
        intensityMin = np.min(rms) if len(rms) > 0 else np.nan
        intensityMax = np.max(rms) if len(rms) > 0 else np.nan
        intensityMean = np.mean(rms) if len(rms) > 0 else np.nan
        intensitySD = np.std(rms) if len(rms) > 0 else np.nan
        intensityQuant = np.percentile(rms, 25) if len(rms) > 0 else np.nan
        diffIntMaxMin = intensityMax - intensityMin if intensityMin and intensityMax else np.nan
        diffIntMaxMean = intensityMax - intensityMean if intensityMax and intensityMean else np.nan
        diffIntMaxMode = intensityMax - intensityQuant if intensityMax and intensityQuant else np.nan
        try:
            snd = parselmouth.Sound(audio_path)
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitterRap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            meanPeriod = parselmouth.praat.call(point_process, "Get mean period",  0, 0, 0.0001, 0.02,1.3)
            pitch = parselmouth.praat.call(snd, "To Pitch", 0.0, 75, 600)
            pitch_values = parselmouth.praat.call(pitch, "To Matrix")
            pitch_data = np.array(pitch_values.values)
            unvoiced_frames = np.sum(pitch_data == 0)
            percentUnvoiced = (unvoiced_frames / pitch_data.size) * 100
            pitch_1 = parselmouth.praat.call(snd, "To Pitch (ac)", 0.0, 75, 15, 0, 0.03, 0.45, 0.01, 0.35, 0.14, 600)
            point_process_1 = parselmouth.praat.call(pitch_1, "To PointProcess")
            max_period = 0.02  
            voice_breaks = 0
            total_periods = 0
            num_points = parselmouth.praat.call(point_process_1, "Get number of points")
            if num_points > 1:  
                for i in range(1, num_points):
                    t1 = parselmouth.praat.call(point_process_1, "Get time from index", i)
                    if i < num_points:  
                        t2 = parselmouth.praat.call(point_process_1, "Get time from index", i+1)
                        period = t2 - t1
                        total_periods += 1
                        if period > (3 * max_period):  
                            voice_breaks += 1
            PercentBreaks = (voice_breaks / total_periods) * 100 if total_periods > 0 else 0
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            numVoiceBreaks = voice_breaks
        except Exception as e:
            print(f"Parselmouth feature extraction failed for {audio_path}: {e}")
            jitter = shimmer = jitterRap = meanPeriod = percentUnvoiced = numVoiceBreaks = PercentBreaks = np.nan

        return {
            "participant": os.path.basename(audio_path),
            "duration": duration,
            "energy": energy,
            "power": power,
            "min_pitch": min_pitch,
            "max_pitch": max_pitch,
            "mean_pitch": mean_pitch,
            "pitch_sd": pitch_sd,
            "pitch_abs": pitch_abs,
            "pitch_quant": pitch_quant,
            "pitchUvsVRatio": pitchUvsVRatio,
            "diffPitchMaxMin": diffPitchMaxMin,
            "diffPitchMaxMean": diffPitchMaxMean,
            "diffPitchMaxMode": diffPitchMaxMode,
            "intensityMin": intensityMin,
            "intensityMax": intensityMax,
            "intensityMean": intensityMean,
            "intensitySD": intensitySD,
            "intensityQuant": intensityQuant,
            "diffIntMaxMin": diffIntMaxMin,
            "diffIntMaxMean": diffIntMaxMean,
            "diffIntMaxMode": diffIntMaxMode,
            "jitter": jitter,
            "shimmer": shimmer,
            "jitterRap": jitterRap,
            "meanPeriod": meanPeriod,
            "percentUnvoiced": percentUnvoiced,
            "numVoiceBreaks": numVoiceBreaks,
            "PercentBreaks": PercentBreaks,
        }

    except Exception as e:
        print(f"Feature extraction failed for {audio_path}: {e}")
        return {col: np.nan for col in ["participant", "duration", "energy", "power", "min_pitch", "max_pitch",
                                        "mean_pitch", "pitch_sd", "pitch_abs", "pitch_quant", "pitchUvsVRatio",
                                        "diffPitchMaxMin", "diffPitchMaxMean", "diffPitchMaxMode", "intensityMin",
                                        "intensityMax", "intensityMean", "intensitySD", "intensityQuant",
                                        "diffIntMaxMin", "diffIntMaxMean", "diffIntMaxMode", "jitter", "shimmer",
                                        "jitterRap", "meanPeriod", "percentUnvoiced", "numVoiceBreaks",
                                        "PercentBreaks"]}


wav_folder = "C:\\Users\\deeps\\Documents\\CSCI 535_ PROJECT\\MAS_dataset\\dataset\\MAS\\wav_folder"
audio_data = []

for file in os.listdir(wav_folder):
    if file.endswith(".wav"):
        path = os.path.join(wav_folder, file)
        audio_data.append(extract_audio_features(path))

df_audio = pd.DataFrame(audio_data)
df_audio.to_csv("audio_features.csv", index=False)
print("Audio features saved to audio_features.csv")
