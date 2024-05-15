import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id='6a26d4cc6399453ab852009763b9ad68', client_secret='3a3878c4636b4d2aa73a1460d656d435')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

tracks = sp.search(q='kendrick lamar', limit=10)
for track in tracks['tracks']['items']:
    print(track['name'], '-', track['artists'][0]['name'])