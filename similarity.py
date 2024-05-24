import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# load data
data = pd.read_csv('updated_data_moods.csv')

# drop unnecessary columns
feature = data.drop(columns=['name', 'album', 'artist', 'id', 'release_date', 'mood'])

# drop any rows with NaN values
feature = feature.dropna()
feature = feature.fillna(0)

# create a scaler object
scaler = StandardScaler()

# fit and transform the data
normalized_features = scaler.fit_transform(feature)

# random index as our test song
random_index = random.randint(0, len(normalized_features) - 1)
test_song = normalized_features[random_index]

# calculate similarity between two feature vectors
def calculate_similarity(features):
    # reshape features1 and features2 to 2D arrays
    max_similarity = -1
    index = 0
    # calculate cosine similarity
    for i in range(0, len(features)):
        if i != random_index:
            similarity = cosine_similarity([normalized_features[i]], [test_song])
            if max_similarity < similarity:
                max_similarity = similarity
                index = i 
    return index

# def calculate_hit_rate(features, test_song_index):
#     # remove test song from dataset
#     features_without_test = np.delete(features, test_song_index, axis=0)
# 
#     # train KNN model
#     knn = NearestNeighbors(n_neighbors=1)
#     knn.fit(features_without_test)
# 
#     # find most similar song
#     _, indices = knn.kneighbors([features[test_song_index]])
#     winner_index = indices[0][0]
# 
#     # check if most similar song is the test song
#     is_accurate = winner_index == test_song_index
# 
#     test_song = f"{data['artist'][test_song_index]} - {data['name'][test_song_index]}" 
#     winner = f"{data['artist'][winner_index]} - {data['name'][winner_index]}" 
# 
#     print(test_song + " vs " + winner + " is " + str(is_accurate))


def calculate_hit_rate(features, data):
    # ensure length of features and data is the same
    if len(features) != len(data):
        raise ValueError('Length of features and data must be the same.')

    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(features, data[['artist', 'name']], test_size=0.2, random_state=42)

    # Convert y_train and y_test back to DataFrames
    y_train = pd.DataFrame(y_train, columns=['artist', 'name'])
    y_test = pd.DataFrame(y_test, columns=['artist', 'name'])

    # train a KNN model on the training set
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X_train)

    # calculate hit rate on the test set
    hits = 0
    for i, test_song in enumerate(X_test):
        _, indices = knn.kneighbors([test_song])
        winner_index = indices[0][0]

        # Retrieve the metadata of the predicted closest song
        winner_artist = y_train.iloc[winner_index]['artist']
        winner_name = y_train.iloc[winner_index]['name']

        # Retrieve the metadata of the test song
        test_artist = y_test.iloc[i]['artist']
        test_name = y_test.iloc[i]['name']

        if winner_artist == test_artist and winner_name == test_name:
            hits += 1

    hit_rate = hits / len(X_test)
    print(f'hit rate: {hit_rate:.2f}')



# test hit rate
calculate_hit_rate(normalized_features, data)
    

# calculate similarity
# winner_index = calculate_similarity(feature)
# 
# print(f"The most similar song to {data['artist'][random_index]} - {data['name'][random_index]} is {data['name'][winner_index]} - {data['artist'][winner_index]}.")
