import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/project/msoleyma_1026/project_Interview/fused_features.csv')

label_cols = [
    'EyeContact', 'SpeakingRate', 'Engaged', 'Paused',
    'Calm', 'NotStressed', 'Focused', 'Authentic', 'NotAwkward'
]

id_col = df['participant'] if 'participant' in df.columns else df.iloc[:, 0]

X = df.drop(columns=label_cols)
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())
X = X.fillna(0)

y = df[label_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X_scaled, y, id_col, test_size=0.2, random_state=42
)

base_model = Ridge()
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse_list = []
for i, col in enumerate(label_cols):
    mse = mean_squared_error(y_test[col], y_pred[:, i])
    print(f"MSE for {col}: {mse:.4f}")
    mse_list.append(mse)

print(f"Average MSE: {np.mean(mse_list):.4f}")
