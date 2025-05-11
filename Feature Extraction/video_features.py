import os
import json
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

clips_dir = '/home1/jainak/MAS_dataset/dataset/MAS/clips'
annotations_file = '/home1/jainak/MAS_dataset/dataset/MAS/annotations.json'
output_csv = '/home1/jainak/MAS_dataset/dataset/MAS/dataset.csv'

with open(annotations_file, 'r') as f:
    annotations = json.load(f)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_features(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    def lm(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])
    def euclidean(p1, p2): return np.linalg.norm(p1 - p2)

    try:
        nose = lm(1)
        chin = lm(152)
        left_eye_outer = lm(33)
        right_eye_outer = lm(263)
        mouth_left = lm(61)
        mouth_right = lm(291)

        image_points = np.array([nose, chin, left_eye_outer, right_eye_outer, mouth_left, mouth_right], dtype='double')
        model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ])
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype='double')
        dist_coeffs = np.zeros((4,1))

        _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = angles
    except:
        pitch = yaw = roll = np.nan

    try:
        inBrL = euclidean(lm(70), lm(159))
        otBrL = euclidean(lm(105), lm(145))
        inBrR = euclidean(lm(336), lm(386))
        otBrR = euclidean(lm(334), lm(374))
        EyeOL = euclidean(lm(159), lm(145))
        EyeOR = euclidean(lm(386), lm(374))
        oLipH = euclidean(lm(13), lm(14))
        iLipH = euclidean(lm(78), lm(308))
        LipCDt = euclidean(lm(61), lm(291))
        dic_indices = [1, 33, 61, 291, 13, 14, 159, 145, 386, 374, 78, 308,
                       70, 105, 336, 334, 66, 107, 10, 152, 263, 362, 133, 386]
        dic_values = np.concatenate([lm(i) for i in dic_indices])
        dic_features = {f'dicCoeff_local{i}': dic_values[i] for i in range(24)}
    except:
        inBrL = otBrL = inBrR = otBrR = EyeOL = EyeOR = oLipH = iLipH = LipCDt = np.nan
        dic_features = {f'dicCoeff_local{i}': np.nan for i in range(24)}

    return {
        'Pitch': pitch, 'Yaw': yaw, 'Roll': roll,
        'inBrL': inBrL, 'otBrL': otBrL, 'inBrR': inBrR, 'otBrR': otBrR,
        'EyeOL': EyeOL, 'EyeOR': EyeOR,
        'oLipH': oLipH, 'iLipH': iLipH, 'LipCDt': LipCDt,
        **dic_features
    }

dataset = []

video_files = sorted(os.listdir(clips_dir))
normalized_files = {f: f.replace('.mp4.mp4', '.mp4') for f in video_files if f.endswith('.mp4.mp4') and not f.startswith('._')}

for original_filename, normalized_filename in normalized_files.items():
    video_path = os.path.join(clips_dir, original_filename)

    if normalized_filename not in annotations:
        continue

    labels = annotations[normalized_filename]
    stress_score = len(labels) - labels.count('no stress')

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        interval = 1
    else:
        interval = max(int(fps), 1)

    frame_idx = 0
    feature_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            feats = extract_features(frame)
            if feats:
                feature_list.append(feats)
        frame_idx += 1

    cap.release()

    if not feature_list:
        avg_features = {
            'Pitch': 0.0, 'Yaw': 0.0, 'Roll': 0.0,
            'inBrL': 0.0, 'otBrL': 0.0, 'inBrR': 0.0, 'otBrR': 0.0,
            'EyeOL': 0.0, 'EyeOR': 0.0, 'oLipH': 0.0, 'iLipH': 0.0, 'LipCDt': 0.0,
            **{f'dicCoeff_local{i}': 0.0 for i in range(24)}
        }
    else:
        avg_features = pd.DataFrame(feature_list).mean().to_dict()

    data_row = {
        'participant': normalized_filename,
        'stress_score': stress_score,
        **avg_features
    }

    dataset.append(data_row)
    print(f"Processed '{normalized_filename}'")

df = pd.DataFrame(dataset)
df.to_csv(output_csv, index=False)