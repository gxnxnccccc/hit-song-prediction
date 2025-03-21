import pandas as pd
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from random_forest_model import HitSongPredictor  

def main():
    print("Loading music dataset...")
    try:
        df = pd.read_csv('songs_filtered.csv')
        print(f"Loaded dataset with {df.shape[0]} songs and {df.shape[1]} features")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print("\nDataset information:")
    print(df.info())
    
    print("\nPopularity statistics:")
    print(f"Min: {df['popularity'].min()}")
    print(f"Max: {df['popularity'].max()}")
    print(f"Mean: {df['popularity'].mean():.2f}")
    print(f"Median: {df['popularity'].median()}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['popularity'], bins=20, alpha=0.7)
    plt.title('Distribution of Song Popularity')
    plt.xlabel('Popularity')
    plt.ylabel('Number of Songs')
    plt.savefig('popularity_distribution.png')
    
    numeric_columns = df.select_dtypes(include=['number']).columns
    correlations = df[numeric_columns].corr()['popularity'].sort_values(ascending=False)
    print("\nFeature correlations with popularity:")
    print(correlations)

    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_columns].corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=90)
    plt.yticks(range(len(numeric_columns)), numeric_columns)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    features = df.select_dtypes(include=['number']).columns.tolist()
    features.remove('popularity')  
    
    null_counts = df[features].isnull().sum()
    print("\nNull values in features:")
    print(null_counts[null_counts > 0])
    
    df['decade'] = (df['year'] // 10) * 10
    
    print("\nInitializing hit song predictor model...")
    predictor = HitSongPredictor(features=features, target='popularity')
    
    start_time = time.time()
    
    training_results = predictor.train(
        df, 
        hyperparameter_tuning=True, 
        use_feature_selection=True
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours):02}:{int(minutes):02}:{seconds:.2f} (hh:mm:ss)")
    print(f"Total seconds: {elapsed_time:.2f}")
    
    training_results['training_time_seconds'] = elapsed_time
    
    print("\nModel regression performance:")
    print(f"RMSE: {training_results['rmse']:.2f}")
    print(f"MAE: {training_results['mae']:.2f}")
    print(f"R² Score: {training_results['r2_score']:.4f}")
    
    with open('training_results.json', 'w') as f:

        results_for_json = {}
        for k, v in training_results.items():
            if isinstance(v, np.floating):
                results_for_json[k] = float(v)
            elif isinstance(v, np.integer):
                results_for_json[k] = int(v)
            elif isinstance(v, dict):
                results_for_json[k] = {key: float(val) if isinstance(val, np.floating) else val 
                                      for key, val in v.items()}
            elif isinstance(v, list):
                results_for_json[k] = v
            else:
                results_for_json[k] = v
        
        json.dump(results_for_json, f, indent=2)
    print("\nTraining results saved to training_results.json")
    
    plt.figure(figsize=(12, 8))
    feature_importance = training_results['feature_importance']
    features_list = list(feature_importance.keys())
    importance_values = list(feature_importance.values())
    
    sorted_idx = np.argsort(importance_values)[::-1]
    
    if len(features_list) > 15:
        sorted_idx = sorted_idx[:15]
    
    plt.barh([features_list[i] for i in sorted_idx], 
            [importance_values[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top Features for Predicting Song Popularity')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    X, y, _ = predictor.preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    predictor.save_model('hit_song_model.pkl')
    
    print("\nTesting model with sample songs...")
    
    high_popularity = df.nlargest(5, 'popularity')
    print("\nHigh Popularity Songs:")
    for _, song in high_popularity.iterrows():
        song_df = pd.DataFrame([song])
        prediction = predictor.predict(song_df)
        print(f"\nSong: {song['song']} by {song['artist']}")
        print(f"Year: {song['year']}")
        print(f"Actual popularity: {song['popularity']}")
        print(f"Predicted score: {prediction['score']:.2f}")
        print("Positive factors:", ", ".join(prediction['positiveFactors'][:2]))
        print("Negative factors:", ", ".join(prediction['negativeFactors'][:2]))
        
        print("\nKey feature values used for prediction:")
        for feature in predictor.selected_features:
            if feature in song_df.columns:
                value = song_df[feature].iloc[0]
                if feature == 'tempo':
                    print(f"  {feature}: {value:.1f} BPM")
                elif feature == 'loudness':
                    print(f"  {feature}: {value:.1f} dB")
                elif feature == 'duration_ms':
                    minutes = int(value / 60000)
                    seconds = int((value % 60000) / 1000)
                    print(f"  {feature}: {minutes}:{seconds:02d}")
                elif feature in ['danceability', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'speechiness']:
                    print(f"  {feature}: {value:.2f}")
                elif feature == 'key':
                    key_names = ['C', 'C♯/D♭', 'D', 'D♯/E♭', 'E', 'F', 'F♯/G♭', 'G', 'G♯/A♭', 'A', 'A♯/B♭', 'B']
                    print(f"  {feature}: {key_names[int(value)]}")
                elif feature == 'mode':
                    print(f"  {feature}: {'Major' if value == 1 else 'Minor'}")
                else:
                    print(f"  {feature}: {value}")
    
    low_popularity = df.nsmallest(5, 'popularity')
    print("\nLow Popularity Songs:")
    for _, song in low_popularity.iterrows():
        song_df = pd.DataFrame([song])
        prediction = predictor.predict(song_df)
        print(f"\nSong: {song['song']} by {song['artist']}")
        print(f"Year: {song['year']}")
        print(f"Actual popularity: {song['popularity']}")
        print(f"Predicted score: {prediction['score']:.2f}")
        print("Positive factors:", ", ".join(prediction['positiveFactors'][:2]))
        print("Negative factors:", ", ".join(prediction['negativeFactors'][:2]))
        
        print("\nKey feature values used for prediction:")
        for feature in predictor.selected_features:
            if feature in song_df.columns:
                value = song_df[feature].iloc[0]

                if feature == 'tempo':
                    print(f"  {feature}: {value:.1f} BPM")
                elif feature == 'loudness':
                    print(f"  {feature}: {value:.1f} dB")
                elif feature == 'duration_ms':
                    minutes = int(value / 60000)
                    seconds = int((value % 60000) / 1000)
                    print(f"  {feature}: {minutes}:{seconds:02d}")
                elif feature in ['danceability', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'speechiness']:
                    print(f"  {feature}: {value:.2f}")
                elif feature == 'key':
                    key_names = ['C', 'C♯/D♭', 'D', 'D♯/E♭', 'E', 'F', 'F♯/G♭', 'G', 'G♯/A♭', 'A', 'A♯/B♭', 'B']
                    print(f"  {feature}: {key_names[int(value)]}")
                elif feature == 'mode':
                    print(f"  {feature}: {'Major' if value == 1 else 'Minor'}")
                else:
                    print(f"  {feature}: {value}")
    
    print("\nEvaluating prediction accuracy on all songs...")
    all_predictions = []
    for _, song in df.iterrows():
        song_df = pd.DataFrame([song])
        prediction = predictor.predict(song_df)
        error = abs(song['popularity'] - prediction['score'])
        
        all_predictions.append({
            'song': song['song'],
            'artist': song['artist'],
            'actual': song['popularity'],
            'predicted': prediction['score'],
            'error': error
        })
    
    predictions_df = pd.DataFrame(all_predictions)
    
    print("\nMost Accurately Predicted Songs:")
    for _, pred in predictions_df.nsmallest(5, 'error').iterrows():
        print(f"{pred['song']} by {pred['artist']}: Actual={pred['actual']}, Predicted={pred['predicted']:.1f}")
    
    print("\nLeast Accurately Predicted Songs:")
    for _, pred in predictions_df.nlargest(5, 'error').iterrows():
        print(f"{pred['song']} by {pred['artist']}: Actual={pred['actual']}, Predicted={pred['predicted']:.1f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(predictions_df['error'], bins=20, alpha=0.7)
    plt.axvline(predictions_df['error'].mean(), color='red', linestyle='--', 
                label=f'Mean error: {predictions_df["error"].mean():.2f}')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Error in Popularity Score')
    plt.ylabel('Number of Songs')
    plt.legend()
    plt.savefig('error_distribution.png')
    plt.close()
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()