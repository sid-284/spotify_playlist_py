import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(
    client_id='nice',
    client_secret='damm'
)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to fetch and store playlist data
def fetch_playlist_data(playlist_id):
    results = spotify.playlist_tracks(playlist_id)
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
    
    return playlist_tracks_data

# Load existing data
file_name = 'spotify_playlist_details.json'
try:
    with open(file_name, 'r') as json_file:
        existing_data = json.load(json_file)
except FileNotFoundError:
    existing_data = []

# Fetch and add data from playlists
playlist1_id = '37i9dQZF1DWU5hC93Ey99S'


existing_data.extend(fetch_playlist_data(playlist1_id))


# Save the updated data back to the JSON file
with open(file_name, 'w') as json_file:
    json.dump(existing_data, json_file, indent=4)

print(f"Data saved to {file_name}")
