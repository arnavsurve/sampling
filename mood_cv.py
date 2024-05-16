import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

# Load data
data = pd.read_csv('data_moods.csv')
X = data[['length', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo', 'key']]
y = data['mood']

# Encode mood labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define parameters for GridSearchCV
params = {
    'hidden_layer_sizes': [(100,), (100, 100), (50, 50, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [500, 2000]
}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize GridSearchCV
grid_search = GridSearchCV(MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, learning_rate_init=0.001, solver='sgd'), params, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Output best parameters
print("Best parameters:", grid_search.best_params_)

# Evaluate the best model found by GridSearchCV
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
