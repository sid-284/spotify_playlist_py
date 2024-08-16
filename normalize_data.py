import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

with open('spotify_playlist_details.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Normalize numerical features
scaler = MinMaxScaler()
df[['tempo_bpm', 'energy', 'danceability', 'popularity']] = scaler.fit_transform(df[['tempo_bpm', 'energy', 'danceability', 'popularity']])
