import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
data = pd.read_csv('data_moods.csv')
X = data[['length', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo', 'key']]
y = data['mood']

# Encode mood labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configure and train the MLP Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)


# Convert numeric predictions back to original mood labels
predicted_moods = le.inverse_transform(y_pred)
actual_moods = le.inverse_transform(y_test)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
comparison_df = pd.DataFrame({'Actual Mood': actual_moods, 'Predicted Mood': predicted_moods})
print(comparison_df)
# Example output of predicted moods
# print(predicted_moods)
