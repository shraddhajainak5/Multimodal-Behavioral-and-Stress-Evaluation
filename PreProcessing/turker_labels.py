import pandas as pd
df = pd.read_csv("/project/msoleyma_1026/MIT_INTERVIEW_DATASET/Labels/turker_scores_full_interview.csv")
df = df[df['Worker'] != 'AGGR']
df = df.drop(columns=['Worker', 'Overall', 'RecommendHiring', 'Colleague', 'Excited', 'Smiled', 'Friendly', 'EngagingTone', 'StructuredAnswers', 'Total'])
df_labels = df.groupby('Participant').mean().reset_index()
df_labels = df_labels.round(2)
df_labels.rename(columns={"Participant": "participant"}, inplace=True)
df_labels.to_csv("/project/msoleyma_1026/project_Interview/final_project/labels_averaged.csv", index=False)
print("Saved cleaned label data to labels_averaged.csv")
