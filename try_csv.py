import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (assuming CSV file)
df = pd.read_csv('data.csv')

# Define features for your model, including acousticness
features = ['tempo', 'energy', 'danceability', 'popularity', 'acousticness']

# Handle NaN values (choose one of the methods below)
# Drop rows with NaN values
df.dropna(subset=features, inplace=True)

# OR
# Fill NaN values with the mean of the column
# df[features] = df[features].fillna(df[features].mean())

# OR
# Fill NaN values with 0
# df[features] = df[features].fillna(0)

# Normalize the selected features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into features and target
X = df[features]
y = df['popularity']  # Example target

# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X, y)

# Evaluate the model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}")

# Recommendation system function
def get_recommendations(input_song_id, df, model, features):
    # Check if the input song is in the dataset
    if input_song_id not in df['id'].values:
        print(f"Song ID {input_song_id} not found in the dataset.")
        return None
    
    # Extract the features of the input song
    input_song_features = df[df['id'] == input_song_id][features]
    
    # Calculate cosine similarity between the input song and all other songs
    similarities = cosine_similarity(input_song_features, df[features])
    
    # Add similarity scores to the DataFrame
    df['similarity_score'] = similarities[0]
    
    # Sort by similarity score and return the top recommendations
    recommendations = df.sort_values(by='similarity_score', ascending=False).head(50)
    return recommendations

# Example song ID for recommendation
input_song_id = '2q95XoeFGixx8b5LNF6Ey1'

# Get recommendations
recommendations = get_recommendations(input_song_id, df, model, features)

# Display the recommendations
if recommendations is not None:
    print(recommendations[['name', 'similarity_score']])
