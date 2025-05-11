import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

data = pd.read_csv('/home1/jainak/MAS_dataset/dataset/MAS/combined_features.csv')
X = data.drop(columns=['participant', 'overall_stress_score'])
y = data['overall_stress_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

selector = SelectFromModel(
    estimator=lgb.LGBMRegressor(
        objective='regression',
        random_state=42,
        num_leaves=31,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=100,
        verbosity=-1
    ),
    threshold='median'
)
selector.fit(X_train_scaled, y_train)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel  = selector.transform(X_test_scaled)

model = lgb.LGBMRegressor(
    objective='regression',
    random_state=42,
    num_leaves=31,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    verbosity=-1
)
model.fit(X_train_sel, y_train)

y_pred = model.predict(X_test_sel)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
