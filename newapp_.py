from flask import Flask, request, jsonify, render_template
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import spacy
import random

# List of random music facts
music_facts = [
    "Taylor Swift claimed all top 10 Billboard Hot 100 spots in 2022.",
    "Adele's '30' was the best-selling album of 2021 worldwide.",
    "Billie Eilish became the youngest artist to headline Glastonbury in 2022.",
    "Drake broke the record for most top 10 Billboard Hot 100 entries in 2021.",
    "Harry Styles' 'As It Was' became the longest-running number-one single of 2022.",
    "Bad Bunny's 'Un Verano Sin Ti' was the most streamed album globally in 2022.",
    "Lil Nas X's 'Old Town Road' spent a record 19 weeks atop the Billboard Hot 100.",
    "Doja Cat's 'Say So' earned her first Billboard Hot 100 number-one hit in 2020.",
    "Justin Bieber's 'Peaches' debuted at number one on the Billboard Hot 100 in 2021.",
    "Dua Lipa's 'Future Nostalgia' won the Grammy for Best Pop Vocal Album in 2021.",
    "Rihanna's Fenty Beauty made her a billionaire, solidifying her status beyond music.",
    "Post Malone broke the record for most weeks in Billboard's Top 10 with 'Circles'.",
    "Ed Sheeran's 'Shape of You' is Spotify's most streamed song with 3 billion+ streams.",
    "Adele's Las Vegas residency sold out within minutes in 2022.",
    "Billie Eilish's 'Happier Than Ever' topped the charts in over 20 countries.",
    "The Rolling Stones celebrated their 60th anniversary in 2022.",
    "Lil Nas X's 'Montero' became an LGBTQ+ anthem, topping charts globally.",
    "This was developed by Jess and Sid."
]


app = Flask(__name__)

# Load the spaCy model for English
nlp = spacy.load('en_core_web_sm')

# Load data from JSON file
with open('spotify_playlist_details.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df = df.drop_duplicates(subset='track_id')
df['tempo_energy_ratio'] = df['tempo_bpm'] / df['energy']

# Normalize numerical features
scaler = MinMaxScaler()
df[['tempo_bpm', 'energy', 'danceability', 'popularity']] = scaler.fit_transform(df[['tempo_bpm', 'energy', 'danceability', 'popularity']])

# Example target variable, replace with your actual target
df['target'] = df['tempo_bpm']  # Replace 'target' with your actual target variable

# Features for the model
X = df[['tempo_bpm', 'energy', 'danceability', 'popularity', 'tempo_energy_ratio']]
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
        track = spotify.track(song_id)
        audio_features = spotify.audio_features(song_id)[0]
        
        if track and audio_features:
            song_data = {
                'track_id': track['id'],
                'track_name': track['name'],
                'tempo_bpm': audio_features.get('tempo', None),
                'energy': audio_features.get('energy', None),
                'danceability': audio_features.get('danceability', None),
                'popularity': track.get('popularity', None),
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
            song_details['tempo_energy_ratio'] = song_details['tempo_bpm'] / song_details['energy']
            song_details_df = pd.DataFrame([song_details])
            song_details_df[['tempo_bpm', 'energy', 'danceability', 'popularity']] = scaler.transform(song_details_df[['tempo_bpm', 'energy', 'danceability', 'popularity']])
            df = pd.concat([df, song_details_df], ignore_index=True)
        else:
            return None

    input_song_features = df[df['track_id'] == input_song_id][features]
    similarities = cosine_similarity(input_song_features, df[features])
    df['similarity_score'] = similarities.flatten()
    recommended_songs = df[df['track_id'] != input_song_id].sort_values(by='similarity_score', ascending=False).head(n_recommendations)
    return recommended_songs

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/random-fact', methods=['GET'])
def random_fact():
    fact = random.choice(music_facts)
    return jsonify({"fact": fact})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = chatbot(user_input)
    return jsonify({"response": response})

def chatbot(user_input):
    doc = nlp(user_input.lower())

    # Determine the intent of the user input
    if any(token.lemma_ == 'search' for token in doc):
        song_name = user_input.replace('search', '').strip()
        response, _ = search_song(song_name)
        return response
    elif any(token.lemma_ == 'track' and token.nbor(1).lemma_ == 'detail' for token in doc):
        track_name = user_input.replace('track details', '').strip()
        track_id = get_track_id(track_name)
        if "No track found" not in track_id:
            return track_details(track_id)
        else:
            return track_id
    elif any(token.lemma_ == 'album' and token.nbor(1).lemma_ == 'detail' for token in doc):
        album_name = user_input.replace('album details', '').strip()
        results = spotify.search(q=album_name, type='album', limit=1)
        if results['albums']['items']:
            album_id = results['albums']['items'][0]['id']
            return album_details(album_id)
        else:
            return "No details found."
    elif any(token.lemma_ == 'artist' and token.nbor(1).lemma_ == 'detail' for token in doc):
        artist_name = user_input.replace('artist details', '').strip()
        results = spotify.search(q=artist_name, type='artist', limit=1)
        if results['artists']['items']:
            artist_id = results['artists']['items'][0]['id']
            return artist_details(artist_id)
        else:
            return "No details found."
    elif any(token.lemma_ == 'recommend' for token in doc):
        track_name = user_input.replace('recommend', '').strip()
        track_id = get_track_id(track_name)
        features = ['tempo_bpm', 'energy', 'danceability', 'popularity', 'tempo_energy_ratio']  # Use tempo_energy_ratio instead of normalized
        recommendations = get_recommendations(track_id, df, model, features)
        if recommendations is not None:
            track_names = recommendations['track_name'].tolist()
            return '\n'.join(track_names)
        else:
            return "No recommendations found."
    else:
        return "I can help you with music searches, track details, album info, and artist info. Try 'search [song name]', 'track details [song name]', 'album details [album name]', 'artist details [artist name]', or 'recommend [song name]'."

if __name__ == '__main__':
    app.run(debug=True)
