import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests

session = requests.Session()
session.verify = False

# Provide your Spotify credentials
client_credentials_manager = SpotifyClientCredentials(
    client_id='ff541e48fa02404b8a93bf3664dafa16',
    client_secret='224c41201a9449aab4ff028987455ed9'
)
#enter code from here.

lz_uri = 'spotify:artist:1Xyo4u8uXC1ZmMpatF05PJ'

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
results = spotify.artist_top_tracks(lz_uri)

for track in results['tracks'][:10]:
    print('track    : ' + track['name'])
    print()
