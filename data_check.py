import json
import pandas as pd

with open('spotify_playlist_details.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(df.isnull().sum())
print(df.describe())
