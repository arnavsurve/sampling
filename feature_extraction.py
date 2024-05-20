import librosa
import numpy as np
# import pandas as pd
# import os
# from termcolor import cprint


def spectral_feature_extraction(y, sr):
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_bandwidth)


def rhythm_feature_extraction(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    mean_beat_frame = np.mean(beat_frames)
    mean_beat_interval = np.mean(np.diff(beat_times))

    return tempo, mean_beat_frame, mean_beat_interval

def harmony_feature_extraction(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)

def timbral_feature_extraction(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return mfccs, zcr

def feature_extraction(filename):
    # load audio file
    y, sr = librosa.load(filename, res_type='kaiser_fast')

    # extract different features
    spectral_features = spectral_feature_extraction(y, sr)
    tempo = rhythm_feature_extraction(y, sr)[0]
    mean_beat_frame = rhythm_feature_extraction(y, sr)[1]
    mean_beat_interval = rhythm_feature_extraction(y, sr)[2]
    chroma_mean = harmony_feature_extraction(y, sr)
    mfccs = timbral_feature_extraction(y, sr)[0]
    zcr = timbral_feature_extraction(y, sr)[1]

    # combine all features into a single feature vector
    features = np.concatenate([spectral_features, tempo, [mean_beat_frame], [mean_beat_interval], chroma_mean, mfccs, [zcr]])
    return features
