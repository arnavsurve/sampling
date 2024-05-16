import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv('data_moods.csv')
X = data[['length', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo', 'key']]
y = data['popularity']

# Define a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=True)),
    ('ridge', Ridge())
])

# Setup the parameter grid
param_grid = {
    'poly__degree': [2, 3, 4],
    'ridge__alpha': [10, 100, 1000, 5000]
}

# Setup the grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)

# Best model
print('Best parameters:', grid_search.best_params_)
print('Best cross-validation MSE:', -grid_search.best_score_)

# Evaluate on the testing set
y_pred_test = grid_search.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print('Test MSE:', test_mse)
