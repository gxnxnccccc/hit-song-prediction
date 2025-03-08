# app.py
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from random_forest_model import HitSongPredictor
import pickle

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'hit_song_model.pkl'  # Path to your saved model

# Global variable to store the model
predictor = None

def load_model():
    """Load the trained model from disk"""
    global predictor
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train the model first.")
    
    # Initialize the model
    predictor = HitSongPredictor()
    predictor.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request"""
    try:
        # Get form data
        form_data = {
            'tempo': float(request.form.get('tempo')),
            'loudness': float(request.form.get('loudness')),
            'valence': float(request.form.get('valence')),
            'energy': float(request.form.get('energy')),
            'danceability': float(request.form.get('danceability')),
            'acousticness': float(request.form.get('acousticness')),
            'instrumentalness': float(request.form.get('instrumentalness')),
            'liveness': float(request.form.get('liveness')),
            'speechiness': float(request.form.get('speechiness')),
            'year': int(request.form.get('year')),
            'duration_ms': int(float(request.form.get('duration_ms')) * 60000),  # Convert to ms
            'key': int(request.form.get('key')),
            'mode': int(request.form.get('mode')),
            'song': request.form.get('song', 'Unknown Song'),
            'artist': request.form.get('artist', 'Unknown Artist'),
        }
        
        # Create a DataFrame with the form data
        song_df = pd.DataFrame([form_data])
        
        # Make prediction
        prediction = predictor.predict(song_df)
        
        # Return the prediction as JSON
        return jsonify({
            'success': True,
            'prediction': {
                'score': round(float(prediction['score']), 1),
                'percentage': round(float(prediction['score']), 1),
                'isHit': prediction['isHit'],
                'positiveFactors': prediction['positiveFactors'],
                'negativeFactors': prediction['negativeFactors']
            }
        })
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/example_songs')
def example_songs():
    """Return example songs to populate the form"""
    examples = [
        {
            'name': 'High Energy Pop Song',
            'song': 'Hit Energy Song',
            'artist': 'Pop Star',
            'tempo': 120,
            'loudness': -5.0,
            'valence': 0.8,
            'energy': 0.85,
            'danceability': 0.75,
            'acousticness': 0.1,
            'instrumentalness': 0.01,
            'liveness': 0.2,
            'speechiness': 0.1,
            'year': 2022,
            'duration_ms': 3.5,
            'key': 7,
            'mode': 1
        },
        {
            'name': 'Acoustic Ballad',
            'song': 'Gentle Melody',
            'artist': 'Indie Artist',
            'tempo': 80,
            'loudness': -12.0,
            'valence': 0.4,
            'energy': 0.3,
            'danceability': 0.35,
            'acousticness': 0.85,
            'instrumentalness': 0.05,
            'liveness': 0.1,
            'speechiness': 0.04,
            'year': 2021,
            'duration_ms': 4.2,
            'key': 0,
            'mode': 0
        },
        {
            'name': 'Electronic Dance Track',
            'song': 'Club Banger',
            'artist': 'DJ Producer',
            'tempo': 128,
            'loudness': -4.0,
            'valence': 0.6,
            'energy': 0.9,
            'danceability': 0.85,
            'acousticness': 0.05,
            'instrumentalness': 0.2,
            'liveness': 0.15,
            'speechiness': 0.08,
            'year': 2023,
            'duration_ms': 3.2,
            'key': 5,
            'mode': 1
        }
    ]
    
    return jsonify(examples)

if __name__ == '__main__':
    # Load the model when the app starts
    load_model()
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0')