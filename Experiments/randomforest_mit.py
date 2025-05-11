import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression, VarianceThreshold
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("/Users/dattateja/Documents/CS535-Multimodal/project dataset/fused_features.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
label_cols = [
    'EyeContact', 'SpeakingRate', 'Engaged', 'Paused',
    'Calm', 'NotStressed', 'Focused', 'Authentic', 'NotAwkward'
]
X = df.drop(columns=['participant'] + label_cols)
y = df[label_cols]
X = X.fillna(0)
selector = VarianceThreshold(threshold=0.0)
X_var = selector.fit_transform(X)
X_var = pd.DataFrame(X_var, columns=X.columns[selector.get_support()])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var)
X_scaled = pd.DataFrame(X_scaled, columns=X_var.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
y_train_combined = y_train.mean(axis=1)
f_vals, p_vals = f_regression(X_train, y_train_combined)
_, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
selected_features = X_train.columns[pvals_corrected < 0.05]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
corr_matrix = X_train_selected.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.6)]
X_train_final = X_train_selected.drop(columns=to_drop)
X_test_final = X_test_selected.drop(columns=to_drop)

models = {}
mse_scores = {}

for col in label_cols:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_final, y_train[col])
    y_pred = model.predict(X_test_final)
    mse = mean_squared_error(y_test[col], y_pred)
    mse_scores[col] = mse

average_mse = np.mean(list(mse_scores.values()))

print(average_mse)
print(mse_scores)