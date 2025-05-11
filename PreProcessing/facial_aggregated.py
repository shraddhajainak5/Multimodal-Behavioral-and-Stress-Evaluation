import os
import pandas as pd

facial_dir = "/project/msoleyma_1026/MIT_INTERVIEW_DATASET/Facial_Features/"
all_face = []

for file in os.listdir(facial_dir):
    if file.endswith(".csv"):
        pid = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(facial_dir, file))
        mean_features = df.mean().to_dict()
        mean_features['participant'] = pid
        all_face.append(mean_features)

df_facial = pd.DataFrame(all_face)
cols = ['participant'] + [col for col in df_facial.columns if col != 'participant']
df_facial = df_facial[cols]
df_facial.to_csv("/project/msoleyma_1026/project_Interview/final_project/facial_features_aggregated.csv", index=False)
