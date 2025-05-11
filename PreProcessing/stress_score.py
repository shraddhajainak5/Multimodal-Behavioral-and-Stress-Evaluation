import os
import json
import pandas as pd
from collections import Counter

clips_folder = "/home1/jainak/MAS_dataset/dataset/MAS/clips"
annotation_file = "/home1/jainak/MAS_dataset/dataset/MAS/annotations.json"
output_csv = "/home1/jainak/MAS_dataset/dataset/MAS/stress_scores.csv"

severity_bonus = [
    ({"body stress", "face stress"}, 1.0),
    ({"audio stress", "prosodic stress"}, 0.5),
    ({"body stress", "face stress", "prosodic stress"}, 1.5),
    ({"face stress", "audio stress"}, 1.0),
    ({"body stress", "audio stress"}, 0.8),
]

with open(annotation_file, "r") as f:
    annotations = json.load(f)

clips_available = sorted(os.listdir(clips_folder))

stress_counter = Counter()

for clip in clips_available:
    if not clip.endswith(".mp4.mp4"):
        continue

    normalized_clip_name = clip.replace('.mp4.mp4', '.mp4')

    if normalized_clip_name not in annotations:
        continue

    labels = annotations[normalized_clip_name]
    stress_counter.update(labels)

total_labels = sum(stress_counter.values())

dynamic_weights = {}
for stress_type, count in stress_counter.items():
    if stress_type == 'no stress':
        dynamic_weights[stress_type] = 0.0
    else:
        dynamic_weights[stress_type] = total_labels / count  

max_weight = max(dynamic_weights.values())
for stress_type in dynamic_weights:
    if dynamic_weights[stress_type] > 0:
        dynamic_weights[stress_type] = (dynamic_weights[stress_type] / max_weight) * 5

rows = []

for clip in clips_available:
    if not clip.endswith(".mp4.mp4"):
        continue

    normalized_clip_name = clip.replace('.mp4.mp4', '.mp4')

    if normalized_clip_name not in annotations:
        continue

    labels = annotations[normalized_clip_name]
    row = {'participant': normalized_clip_name}

    present_stresses = set(labels)

    for stress_type in dynamic_weights.keys():
        row[stress_type] = dynamic_weights.get(stress_type, 0.0) if stress_type in present_stresses else 0.0

    total_score = sum(row[stress_type] for stress_type in dynamic_weights.keys())

    for required_stresses, bonus in severity_bonus:
        if required_stresses.issubset(present_stresses):
            total_score += bonus

    row['raw_score'] = round(total_score, 4)
    rows.append(row)

df = pd.DataFrame(rows)

float_cols = df.select_dtypes(include=['float']).columns
df[float_cols] = df[float_cols].round(3)

df['overall_stress_score'] = 0.0

stressed_mask = df[['audio stress', 'body stress', 'face stress', 'prosodic stress']].sum(axis=1) > 0

if stressed_mask.sum() > 0:
    stressed_scores = df.loc[stressed_mask, 'raw_score']
    min_score = stressed_scores.min()
    max_score = stressed_scores.max()

    df.loc[stressed_mask, 'overall_stress_score'] = (
        1 + (stressed_scores - min_score) / (max_score - min_score) * 6
    ).round(3)

df.drop(columns=['raw_score'], inplace=True)

df.to_csv(output_csv, index=False)
