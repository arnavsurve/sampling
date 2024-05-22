from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv('updated_data_moods.csv')

# Drop unnecessary columns
# Drop unnecessary columns
feature = data.drop(columns=['name', 'album', 'artist', 'id', 'release_date', 'mood'])

# Drop any rows with NaN values
feature = feature.dropna()
feature = feature.fillna(0)

# Create a scaler object
scaler = StandardScaler()

# Fit and transform the data
normalized_feature = scaler.fit_transform(feature)
#random index as our test song
random_index = random.randint(0, len(normalized_feature))
test_song = normalized_feature[random_index]


# calculate similarity between two feature vectors
def calculate_similarity(features):

    # reshape features1 and features2 to 2D arrays
    max_similarity = -1
    index = 0
    # calculate cosine similarity
    for i in range(0, len(features)):
        if i != random_index:
            similarity = cosine_similarity([normalized_feature[i]], [test_song])
            if max_similarity < similarity:
                max_similarity = similarity
                index = i 
    return index

    

song1_filename = './samples/23.mp3'
song2_filename = './samples/1999.mp3'


# calculate similarity
winner_index = calculate_similarity(feature)
winner = data['name'][winner_index]

print(f"The most similar song to {data['name'][random_index]} by {data['artist'][random_index]} is {winner} by {data['artist'][winner_index]}")
