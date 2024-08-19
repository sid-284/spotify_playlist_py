import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

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

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Initialize Spotify client with credentials
client_credentials_manager = SpotifyClientCredentials(
    client_id='ff541e48fa02404b8a93bf3664dafa16',
    client_secret='224c41201a9449aab4ff028987455ed9'
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

input_song_id = '5ygDXis42ncn6kYG14lEVG'  # Replace with actual track_id
features = ['tempo_bpm', 'energy', 'danceability', 'popularity', 'tempo_energy_ratio_normalized']

recommendations = get_recommendations(input_song_id, df, model, features)
if recommendations is not None:
    print(recommendations[['track_name', 'similarity_score']])
