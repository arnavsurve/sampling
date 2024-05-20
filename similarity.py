from sklearn.metrics.pairwise import cosine_similarity
import feature_extraction

# calculate similarity between two feature vectors
def calculate_similarity(features1, features2):
    # Reshape features1 and features2 to 2D arrays
    features1 = features1.reshape(1, -1)
    features2 = features2.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(features1, features2)
    
    return similarity[0][0]

song1_filename = './samples/23.mp3'
song2_filename = './samples/1999.mp3'

# Extract features from two songs
features1 = feature_extraction.feature_extraction(song1_filename)
features2 = feature_extraction.feature_extraction(song2_filename)

# Calculate similarity
similarity = calculate_similarity(features1, features2)

print(f"The similarity between the two songs is {similarity}")
