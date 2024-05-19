import os
import re
import pandas as pd

from termcolor import cprint

from pytube import YouTube
from pytube.cli import on_progress
from pytube.exceptions import PytubeError

from youtube_search import YoutubeSearch

from pydub import AudioSegment

# search for a song on youtube and get the first result's URL
def search_youtube(song_name):
    results = YoutubeSearch(song_name, max_results=1).to_dict()
    if results:
        video_id = results[0]['id']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        return video_url
    return None

# download audio from a youtube URL and convert it to MP3
def download(url, song_name, output_path='samples'):
    yt = YouTube(url, on_progress_callback=on_progress)
    stream = yt.streams.filter(only_audio=True).first()
    download_path = stream.download(output_path=output_path)

    if '/' in song_name:
        print(f"Skipping download for {song_name} due to invalid character '/' in name. Please download manually.")
        return None

    try:
        # base, ext = os.path.splitext(download_path)
        # mp3_path = base + '.mp3'
        mp3_path = os.path.join(output_path, f'{song_name}.mp3')

        audio = AudioSegment.from_file(download_path)
        audio.export(mp3_path, format='mp3')

        os.remove(download_path)

        return mp3_path

    except PytubeError:
        print(f"error downloading {url}")
        os.remove(download_path)


# load song names from dataset
dataset = input("enter dataset (.csv) to read from: ")
song_df = pd.read_csv(dataset)

# ensure the correct column names
if 'name' not in song_df.columns or 'artist' not in song_df.columns:
    raise KeyError("The CSV file must contain 'name' and 'artist' columns.")

songs_and_artists = song_df[['name', 'artist']].values

# process each song
for song, artist in songs_and_artists:
    filepath = f"./samples/{song}.mp3"
    if os.path.exists(filepath):
        print(f'{filepath} already exists, skipping')
        continue
    else:
        query = f'{song} - {artist}'
        cprint(f'searching for: {query}', 'cyan')
        url = search_youtube(query)
        if url:
            print(f'found URL: {url}')
            mp3_path = download(url, song, 'samples/')
            if mp3_path:
                cprint(f'downloaded and saved as: {mp3_path}', 'green')
            else:
                cprint(f'failed to download {query} - ({url})', 'red')

        else:
            cprint(f'no results found for {song}!', 'red')
