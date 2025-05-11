import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('C:\\Users\\deeps\\Documents\\CSCI 535_ PROJECT\\MAS_dataset\\combined_features.csv')


drop_cols = [
    'participant',
    'Joy', 'Sadness', 'Tentative', 'Analytical', 'Fear', 'Anger',
    'no stress', 'audio stress', 'face stress', 'prosodic stress', 'body stress',
]

X = df.drop(columns=drop_cols)
y = df['overall_stress_score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Random Forest': RandomForestRegressor(random_state=42)
}


param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2', None]
    }
}


for name, model in models.items():
    print(f"Training and tuning: {name}")
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[name],
        n_iter=30,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters for {name}: {random_search.best_params_}")
    print(f"Test MSE ({name}): {mse:.4f}")
    print(f"Test R² Score ({name}): {r2:.4f}")
    print("-" * 50)
