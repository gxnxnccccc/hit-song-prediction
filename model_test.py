from random_forest_model import HitSongPredictor
import pandas as pd

# Initialize the predictor with only numerical features
predictor = HitSongPredictor(features=[
    'tempo', 'loudness', 'valence', 'energy', 
    'danceability', 'acousticness', 'instrumentalness', 
    'liveness', 'speechiness'
])

# Load the trained model
predictor.load_model("hit_song_model.pkl")

# Check if the model and scaler were loaded correctly
print(f"Model loaded from hit_song_model.pkl")
print(f"Model trained on {len(predictor.features)} features")
if hasattr(predictor.scaler, 'feature_names_in_'):
    print(f"Model expects {len(predictor.scaler.feature_names_in_)} features after preprocessing")

print("///////////////////////////////")

# Sample song without genre
sample_song = {
    "tempo": 120,
    "loudness": -5,
    "valence": 0.7,
    "energy": 0.8,
    "danceability": 0.75,
    "acousticness": 0.1,
    "instrumentalness": 0.0,
    "liveness": 0.2,
    "speechiness": 0.05
}

# Convert the dictionary to a DataFrame
sample_df = pd.DataFrame([sample_song])

# Print sample data for verification
print("Sample song data:")
print(sample_df)
print()

try:
    # Make a prediction
    prediction = predictor.predict(sample_df)
    
    # Print the prediction result
    print(f"Predicted popularity score: {prediction['score']:.2f}")
    print(f"Is this a hit song? {'Yes' if prediction['isHit'] else 'No'}")
    print("\nPositive factors:")
    for factor in prediction['positiveFactors']:
        print(f"- {factor}")
    print("\nNegative factors:")
    for factor in prediction['negativeFactors']:
        print(f"- {factor}")
        
except Exception as e:
    print(f"Error during prediction: {str(e)}")