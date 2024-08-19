import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity


# Load data from JSON file
with open('spotify_playlist_details.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df = df.drop_duplicates(subset='track_id')
# Create a new feature: tempo_energy_ratio
df['tempo_energy_ratio'] = df['tempo_bpm'] / df['energy']
df['tempo_energy_ratio_normalized'] = df['tempo_energy_ratio'] / df['tempo_energy_ratio'].max()

# Normalize numerical features
scaler = MinMaxScaler()
df[['tempo_bpm', 'energy', 'danceability', 'popularity']] = scaler.fit_transform(df[['tempo_bpm', 'energy', 'danceability', 'popularity']])

# Example target variable, replace with your actual target
df['target'] = df['tempo_bpm']  # Replace 'target' with your actual target variable

# Features for the model
X = df[['tempo_bpm', 'energy', 'danceability', 'popularity', 'tempo_energy_ratio_normalized']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}")

 # Initialize Spotify client with credentials
client_credentials_manager = SpotifyClientCredentials(
     client_id='dam',
     client_secret='nice'
)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def fetch_song_details_from_api(song_id):
    try:
        # Fetch track details from Spotify
        track = spotify.track(song_id)
        # Fetch audio features for the track
        audio_features = spotify.audio_features(song_id)[0]  # [0] because it returns a list
        
        if track and audio_features:
            # Extract relevant details and return as a dictionary
            song_data = {
                'track_id': track['id'],
                'track_name': track['name'],
                'tempo_bpm': audio_features.get('tempo', None),
                'energy': audio_features.get('energy', None),
                'danceability': audio_features.get('danceability', None),
                'popularity': track.get('popularity', None),
                # Add any other features you need
            }
            return song_data
        else:
            print(f"Failed to fetch details for song ID '{song_id}'.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_recommendations(input_song_id, df, model, features, n_recommendations=50):
    if input_song_id not in df['track_id'].values:
        print(f"Song with track_id '{input_song_id}' not found in the database. Fetching from Spotify API...")
        song_details = fetch_song_details_from_api(input_song_id)
        if song_details:
            # Normalize the new song's features
            song_details['tempo_energy_ratio'] = song_details['tempo_bpm'] / song_details['energy']
            song_details['tempo_energy_ratio_normalized'] = song_details['tempo_energy_ratio'] / df['tempo_energy_ratio'].max()
            song_details_df = pd.DataFrame([song_details])
            song_details_df[['tempo_bpm', 'energy', 'danceability', 'popularity']] = scaler.transform(song_details_df[['tempo_bpm', 'energy', 'danceability', 'popularity']])
            # Append to the DataFrame
            df = pd.concat([df, song_details_df], ignore_index=True)
        else:
            return None

    # Extract features of the input song
    input_song_features = df[df['track_id'] == input_song_id][features]

    # Calculate similarity between input song and all other songs
    similarities = cosine_similarity(input_song_features, df[features])

    # Add similarity scores to DataFrame
    df['similarity_score'] = similarities.flatten()

    # Sort by similarity score and get top recommendations, excluding the input song itself
    recommended_songs = df[df['track_id'] != input_song_id].sort_values(by='similarity_score', ascending=False).head(n_recommendations)

    return recommended_songs



client_credentials_manager = SpotifyClientCredentials(
    client_id="db337ae49ec747539c71ffd8aabbb2e8",
    client_secret="bb5820a911bc4cb0a5e9469543a14073"
)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def search_song(song_name):
    try:
        results = spotify.search(q=song_name, type='track', limit=5)
        tracks = results['tracks']['items']
        
        if tracks:
            response = "Here are the top 5 search results:\n"
            for idx, track in enumerate(tracks, 1):
                response += f"{idx}. {track['name']} by {', '.join(artist['name'] for artist in track['artists'])} (ID: {track['id']})\n"
            return response, [track['id'] for track in tracks]
        else:
            return "Sorry, I couldn't find any songs matching that search.", []
    except Exception as e:
        return f"An error occurred: {str(e)}", []

def track_details(track_id):
    try:
        track = spotify.track(track_id)
        return (f"Track: {track['name']}\n"
                f"Artist(s): {', '.join(artist['name'] for artist in track['artists'])}\n"
                f"Album: {track['album']['name']}\n"
                f"Release Date: {track['album']['release_date']}\n"
                f"Duration: {track['duration_ms'] // 1000} seconds\n"
                f"Popularity: {track['popularity']}\n"
                f"Track ID: {track['id']}")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def album_details(album_id):
    try:
        album = spotify.album(album_id)
        track_list = "\n".join([track['name'] for track in album['tracks']['items']])
        return (f"Album: {album['name']}\n"
                f"Artist: {', '.join(artist['name'] for artist in album['artists'])}\n"
                f"Release Date: {album['release_date']}\n"
                f"Total Tracks: {album['total_tracks']}\n"
                f"Tracks:\n{track_list}")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def artist_details(artist_id):
    try:
        artist = spotify.artist(artist_id)
        top_tracks = spotify.artist_top_tracks(artist_id)
        top_tracks_list = "\n".join([f"{track['name']} - {track['album']['name']}" for track in top_tracks['tracks']])
        return (f"Artist: {artist['name']}\n"
                f"Genres: {', '.join(artist['genres'])}\n"
                f"Popularity: {artist['popularity']}\n"
                f"Top Tracks:\n{top_tracks_list}")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_track_id(track_name):
    try:
        results = spotify.search(q=track_name, type='track', limit=1)
        if results['tracks']['items']:
            track_id = results['tracks']['items'][0]['id']
            return track_id
        else:
            return "No track found with that name."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def chatbot():
    print("Hello! I’m your music chatbot. You can search for music, get track details, album info, and artist info.")
    while True:
        user_input = input("You: ")
        try:
            if 'search' in user_input.lower():
                song_name = user_input.replace('search', '').strip()
                response, track_ids = search_song(song_name)
                print("Bot:", response)
            elif 'track details' in user_input.lower():
                track_name = user_input.replace('track details', '').strip()
                track_id = get_track_id(track_name)
                if "No track found" not in track_id:
                    response = track_details(track_id)
                    print("Bot:", response)
                else:
                    print("Bot:", track_id)
            elif 'album details' in user_input.lower():
                album_name = user_input.replace('album details', '').strip()
                results = spotify.search(q=album_name, type='album', limit=1)
                if results['albums']['items']:
                    album_id = results['albums']['items'][0]['id']
                    response = album_details(album_id)
                    print("Bot:", response)
                else:
                    print("Bot: No details found.")
            elif 'artist details' in user_input.lower():
                artist_name = user_input.replace('artist details', '').strip()
                results = spotify.search(q=artist_name, type='artist', limit=1)
                if results['artists']['items']:
                    artist_id = results['artists']['items'][0]['id']
                    response = artist_details(artist_id)
                    print("Bot:", response)
                else:
                    print("Bot: No details found.")
            # elif 'track id' in user_input.lower():
            #     track_name = user_input.replace('track id', '').strip()
            #     track_id = get_track_id(track_name)
            #     print("Bot: Track ID:", track_id)
            elif 'recommend' in user_input.lower():
                track_name = user_input.replace('track id', '').strip()
                track_id = get_track_id(track_name)
                input_song_id = track_id  # Replace with actual track_id
                features = ['tempo_bpm', 'energy', 'danceability', 'popularity', 'tempo_energy_ratio_normalized']

                recommendations = get_recommendations(input_song_id, df, model, features)
                if recommendations is not None:
                    print(recommendations[['track_name', 'similarity_score']])
                # print("Bot: Track ID:", track_id)
            elif 'exit' in user_input.lower():
                print("Bot: Goodbye!")
                break
            else:
                print("Bot: I can help you with music searches, track details, album info, and artist info. Try 'search [song name]', 'track details [song name]', 'album details [album name]', 'artist details [artist name]', or 'track id [song name]'.")
        except Exception as e:
            print(f"Bot: An error occurred: {str(e)}")


chatbot()
