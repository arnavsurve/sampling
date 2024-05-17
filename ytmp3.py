import os
import re
import pandas as pd

from termcolor import cprint

from pytube import YouTube
from pytube.cli import on_progress
from pytube.exceptions import PytubeError

from youtube_search import YoutubeSearch

from pydub import AudioSegment

def search_youtube(song_name):
    results = YoutubeSearch(song_name, max_results=1).to_dict()
    if results:
        video_id = results[0]['id']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        return video_url
    return None

def download(url, output_path):
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        stream = yt.streams.filter(only_audio=True).first()
        download_path = stream.download(output_path=output_path)

        base, ext = os.path.splitext(download_path)
        mp3_path = base + '.mp3'

        audio = AudioSegment.from_file(download_path)
        audio.export(mp3_path, format='mp3')

        os.remove(download_path)

        return mp3_path

            
    except PytubeError:
        print(f"error downloading {url}: {e}")


# load song names from dataset
dataset = input("enter dataset (.csv) to read from: ")
song_df = pd.read_csv(dataset)
column = input("enter column to read names from: ")
song_names = song_df['name'].tolist()

# process each song
for song in song_names:
    cprint(f'searching for: {song}', 'cyan')
    url = search_youtube(song)
    if url:
        print(f'found URL: {url}')
        mp3_path = download(url, output_path='samples')
        if mp3_path:
            cprint(f'downloaded and saved as: {mp3_path}', 'green')
        else:
            cprint(f'failed to download {song} - ({url})', 'red')

    else:
        cprint(f'no results found for {song}!', 'red')
