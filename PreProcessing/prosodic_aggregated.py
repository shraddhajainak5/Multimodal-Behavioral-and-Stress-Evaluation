import pandas as pd
df = pd.read_csv("/project/msoleyma_1026/MIT_INTERVIEW_DATASET/Prosody/prosodic_features.csv")
df['participant'] = df['participant&question'].str.extract(r'(P\d+)', expand=False)
df = df.drop(columns=['participant&question'])
for col in df.columns:
    if col != 'participant':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df_prosodic = df.groupby('participant').mean(numeric_only=True).reset_index()
df_prosodic.to_csv("/project/msoleyma_1026/project_Interview/final_project/prosodic_features_aggregated.csv", index=False)