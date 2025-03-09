# app.py
import os
import pickle
import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request, jsonify
from random_forest_model import HitSongPredictor

app = Flask(__name__)

@app.route('/test')
def test():
    return "This is a test route!"

# Load the trained model
MODEL_PATH = 'hit_song_model.pkl'
predictor = HitSongPredictor()
predictor.load_model(MODEL_PATH)

# Load the dataset for pre-existing songs
dataset = pd.read_csv('songs_filtered.csv')

# Get feature ranges for sliders
feature_ranges = {}
for feature in predictor.selected_features:
    if feature in dataset.columns and feature not in ['year', 'key', 'mode']:
        feature_ranges[feature] = {
            'min': float(dataset[feature].min()),
            'max': float(dataset[feature].max()),
            'default': float(dataset[feature].median())
        }

# Handle special cases
feature_ranges['year'] = {'min': 1960, 'max': 2023, 'default': 2020}
feature_ranges['key'] = {'min': 0, 'max': 11, 'default': 5}  # Musical keys (C to B)
feature_ranges['mode'] = {'min': 0, 'max': 1, 'default': 1}  # Major (1) or minor (0)

# Features that need special formatting or explanations
feature_explanations = {
    'tempo': 'Beats per minute. Pop hits often range from 100-130 BPM.',
    'loudness': 'Measured in dB, with 0 being max. Commercial tracks are often -8 to -4 dB.',
    'valence': 'Musical positiveness from 0.0 to 1.0. High values sound more positive.',
    'energy': 'Intensity and activity from 0.0 to 1.0. High-energy tracks feel fast and loud.',
    'danceability': 'How suitable for dancing from 0.0 to 1.0, based on tempo, rhythm, beat strength.',
    'acousticness': 'Confidence the track is acoustic from 0.0 to 1.0. 1.0 means high confidence.',
    'instrumentalness': 'Predicts if a track has no vocals. Values above 0.5 are instrumental.',
    'liveness': 'Detects audience in the recording. Higher values mean more likely performed live.',
    'speechiness': 'Presence of spoken words. Values above 0.66 are likely spoken-word tracks.',
    'year': 'Year of release. Recent songs tend to rank higher in popularity algorithms.',
    'duration_ms': 'Length in milliseconds. Hit songs are often between 3 and 4 minutes.',
    'key': 'Musical key (0=C, 1=C♯/D♭, 2=D, etc.)',
    'mode': 'Musical mode (0=minor, 1=major)'
}

# Create a list of sample songs (50 random songs from the dataset)
sample_songs = dataset.sample(n=50).sort_values('popularity', ascending=False)
sample_songs_list = []
for _, row in sample_songs.iterrows():
    sample_songs_list.append({
        'id': int(_),
        'artist': row['artist'],
        'song': row['song'],
        'year': int(row['year']),
        'popularity': int(row['popularity'])
    })

@app.route('/')
def index():
    return render_template('index.html', 
                          feature_ranges=feature_ranges,
                          feature_explanations=feature_explanations,
                          sample_songs=sample_songs_list,
                          features=list(feature_ranges.keys()),
                          feature_importance=predictor.feature_importance)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Check if we're predicting from a sample song or custom values
        if 'songId' in data and data['songId']:
            song_id = int(data['songId'])
            song_data = dataset.iloc[song_id].to_dict()
            prediction = predictor.predict(song_data)
            
            # Add song info to the response
            song_info = {
                'artist': str(song_data['artist']),
                'song': str(song_data['song']),
                'year': int(song_data['year']),
                'actual_popularity': int(song_data['popularity'])
            }
            prediction.update(song_info)
            
            # Add feature values for display
            features = {}
            for feature in predictor.selected_features:
                if feature in song_data:
                    features[feature] = float(song_data[feature])
            prediction['features'] = features
            
        else:
            # Create a song data dictionary from custom values
            song_data = {}
            for feature, value in data['features'].items():
                song_data[feature] = float(value)
                
            prediction = predictor.predict(song_data)
            prediction['features'] = song_data
        
        # Make sure all values are JSON serializable
        safe_prediction = {}
        for key, value in prediction.items():
            if key == 'positiveFactors' or key == 'negativeFactors':
                safe_prediction[key] = [str(item) for item in value]
            elif isinstance(value, (int, float, bool, str, list, dict)) or value is None:
                safe_prediction[key] = value
            else:
                safe_prediction[key] = str(value)
                
        return jsonify(safe_prediction)
    except Exception as e:
        # Return error message
        return jsonify({'error': str(e)}), 500

@app.route('/get_song_features/<int:song_id>')
def get_song_features(song_id):
    song_data = dataset.iloc[song_id].to_dict()
    features = {}
    for feature in predictor.selected_features:
        if feature in song_data:
            features[feature] = float(song_data[feature])
    return jsonify(features)

@app.route('/feature_importance')
def get_feature_importance():
    return jsonify(predictor.feature_importance)

if __name__ == '__main__':
    app.run(debug=True)