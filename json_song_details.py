import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import time
from requests.exceptions import ReadTimeout

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(
    client_id='nice',
    client_secret='dam'
)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Fetch details for each track in the playlist
################################################################################################
#playlist1
playlist_id = '2ZdK1m79Ohvm5za8cLj8yF'  # Replace with your playlist ID
results = spotify.playlist_tracks(playlist_id)

# Store playlist tracks data
playlist_tracks_data = []
for item in results['items']:
    track = item['track']
    track_id = track['id']
    audio_features = spotify.audio_features(track_id)[0]

    track_data = {
        'track_name': track['name'],
        'track_id': track_id,
        'tempo_bpm': audio_features['tempo'],
        'key': audio_features['key'],
        'energy': audio_features['energy'],
        'danceability': audio_features['danceability'],
        'popularity': track['popularity'],
        'album_name': track['album']['name'],
        'release_date': track['album']['release_date']
    }
    playlist_tracks_data.append(track_data)

# Save the data as JSON
file_name = 'spotify_playlist_details.json'
with open(file_name, 'w') as json_file:
    json.dump(playlist_tracks_data, json_file, indent=4)

print(f"Data1 saved to {file_name}")
################################################################################################
#playlist2
playlist_id = '2ZdK1m79Ohvm5za8cLj8yF'  # Replace with your playlist ID
results = spotify.playlist_tracks(playlist_id)

# Store playlist tracks data
playlist_tracks_data = []
for item in results['items']:
    track = item['track']
    track_id = track['id']
    audio_features = spotify.audio_features(track_id)[0]

    track_data = {
        'track_name': track['name'],
        'track_id': track_id,
        'tempo_bpm': audio_features['tempo'],
        'key': audio_features['key'],
        'energy': audio_features['energy'],
        'danceability': audio_features['danceability'],
        'popularity': track['popularity'],
        'album_name': track['album']['name'],
        'release_date': track['album']['release_date']
    }
    playlist_tracks_data.append(track_data)

# Save the data as JSON
file_name = 'spotify_playlist_details.json'
with open(file_name, 'a') as json_file:
    json.dump(playlist_tracks_data, json_file, indent=4)

print(f"Data2 saved to {file_name}")
