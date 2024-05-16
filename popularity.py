import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data_moods.csv')

X = data[['length', 'danceability', 'acousticness', 'energy']]  
y = data['popularity']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

new_song_features = [[300000, 0.7, 0.1, 0.8]]  
predicted_popularity = model.predict(new_song_features)
print('Predicted Popularity:', predicted_popularity[0])
