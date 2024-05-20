import pandas as pd
import os
from termcolor import cprint

import feature_extraction

csv_path = './data_moods.csv'
df = pd.read_csv(csv_path)

# Initialize list to store features
features_list = []
directory = './samples/'
audio_files = sorted(os.listdir(directory))

for audio in audio_files:
    if audio.endswith('.mp3'):
        audio_path = os.path.join(directory, audio)
        cprint(f'Extracting feature vector for {audio_path}', "cyan")
        features = feature_extraction.feature_extraction(audio_path)
        features_list.append(features)

# Adjust the number of columns based on actual features extracted
feature_names = ['Spectral Centroid', 'Spectral Rolloff', 'Spectral Bandwidth', 'Tempo', 'Mean Beat Frame', 'Mean Beat Interval'] + \
                ['Chroma Mean' + str(i) for i in range(12)] + \
                ['MFCC' + str(i) for i in range(13)] + ['ZCR']

# Create DataFrame from the features list
features_df = pd.DataFrame(features_list, columns=feature_names)

# Concatenate original DataFrame with the features DataFrame
df_merged = pd.concat([df, features_df], axis=1)

# Save the merged DataFrame
output_csv_path = 'updated_data_moods.csv'
df_merged.to_csv(output_csv_path, index=False)

cprint(f'Updated dataset written to {output_csv_path}', "green")
