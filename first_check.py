import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data from JSON file
with open('spotify_playlist_details.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Create a new feature: tempo_energy_ratio
df['tempo_energy_ratio'] = df['tempo_bpm'] / df['energy']
df['tempo_energy_ratio_normalized'] = df['tempo_energy_ratio'] / df['tempo_energy_ratio'].max()

# Normalize numerical features
scaler = MinMaxScaler()
df[['tempo_bpm', 'energy', 'danceability', 'popularity']] = scaler.fit_transform(df[['tempo_bpm', 'energy', 'danceability', 'popularity']])
# Example target variable, replace with your actual target
df['target'] = df['tempo_bpm']  # Replace 'popularity' with your target variable

# Features for the model
X = df[['tempo_bpm', 'energy', 'danceability', 'popularity', 'tempo_energy_ratio_normalized']]
y = df['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

# Create the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(input_song_id, df, model, features, n_recommendations=50):
    if input_song_id not in df['track_id'].values:
        print(f"Song with track_id '{input_song_id}' not found in the database.")
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

input_song_id = '2lJn77IeZAP9cmv6DPXOrL'  # Replace with actual track_id
features = ['tempo_bpm', 'energy', 'danceability', 'popularity', 'tempo_energy_ratio_normalized']

recommendations = get_recommendations(input_song_id, df, model, features)
print(recommendations[['track_name', 'similarity_score']])

