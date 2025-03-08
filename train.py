import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from random_forest_model import HitSongPredictor  # Make sure to use the correct import

def main():
    # Load your dataset
    print("Loading music dataset...")
    try:
        df = pd.read_csv('songs_filtered.csv')
        print(f"Loaded dataset with {df.shape[0]} songs and {df.shape[1]} features")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Display dataset info
    print("\nDataset information:")
    print(df.info())
    
    # Analyze target distribution
    print("\nPopularity statistics:")
    print(f"Min: {df['popularity'].min()}")
    print(f"Max: {df['popularity'].max()}")
    print(f"Mean: {df['popularity'].mean():.2f}")
    print(f"Median: {df['popularity'].median()}")
    
    # Examine distribution of popularity
    plt.figure(figsize=(10, 6))
    plt.hist(df['popularity'], bins=20, alpha=0.7)
    plt.title('Distribution of Song Popularity')
    plt.xlabel('Popularity')
    plt.ylabel('Number of Songs')
    plt.savefig('popularity_distribution.png')
    
    # Analyze correlations with popularity
    numeric_columns = df.select_dtypes(include=['number']).columns
    correlations = df[numeric_columns].corr()['popularity'].sort_values(ascending=False)
    print("\nFeature correlations with popularity:")
    print(correlations)
    
    # Plot correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_columns].corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=90)
    plt.yticks(range(len(numeric_columns)), numeric_columns)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Select all numeric features
    features = df.select_dtypes(include=['number']).columns.tolist()
    features.remove('popularity')  # Remove target from features
    
    # Check data quality
    null_counts = df[features].isnull().sum()
    print("\nNull values in features:")
    print(null_counts[null_counts > 0])
    
    # Create year bins to help the model understand temporal trends better
    df['decade'] = (df['year'] // 10) * 10
    
    # Initialize and train the model
    print("\nInitializing hit song predictor model...")
    predictor = HitSongPredictor(features=features, target='popularity')
    
    # Train with hyperparameter tuning and feature selection
    print("\nTraining model with hyperparameter tuning and feature selection (this may take a while)...")
    training_results = predictor.train(df, hyperparameter_tuning=True, use_feature_selection=True)
    
    # Display model performance
    print("\nModel performance:")
    print(f"RMSE: {training_results['rmse']:.2f}")
    print(f"MAE: {training_results['mae']:.2f}")
    print(f"R² Score: {training_results['r2_score']:.4f}")
    
    # Save training results
    with open('training_results.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        results_for_json = {}
        for k, v in training_results.items():
            if isinstance(v, np.floating):
                results_for_json[k] = float(v)
            elif isinstance(v, np.integer):
                results_for_json[k] = int(v)
            elif isinstance(v, dict):
                # Handle feature importance dictionary
                results_for_json[k] = {key: float(val) if isinstance(val, np.floating) else val 
                                      for key, val in v.items()}
            elif isinstance(v, list):
                results_for_json[k] = v
            else:
                results_for_json[k] = v
        
        json.dump(results_for_json, f, indent=2)
    print("\nTraining results saved to training_results.json")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    feature_importance = training_results['feature_importance']
    features_list = list(feature_importance.keys())
    importance_values = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance_values)[::-1]
    
    # Plot only top 15 features if there are many
    if len(features_list) > 15:
        sorted_idx = sorted_idx[:15]
    
    plt.barh([features_list[i] for i in sorted_idx], 
            [importance_values[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top Features for Predicting Song Popularity')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save the trained model
    predictor.save_model('hit_song_model.pkl')
    
    # Evaluate on test samples
    print("\nTesting model with sample songs...")
    
    # Test with high popularity songs
    high_popularity = df.nlargest(5, 'popularity')
    print("\nHigh Popularity Songs:")
    for _, song in high_popularity.iterrows():
        song_df = pd.DataFrame([song])
        prediction = predictor.predict(song_df)
        print(f"\nSong: {song['song']} by {song['artist']}")
        print(f"Year: {song['year']}")
        print(f"Actual popularity: {song['popularity']}")
        print(f"Predicted score: {prediction['score']:.2f}")
        print(f"Is a hit: {prediction['isHit']}")
        print("Positive factors:", ", ".join(prediction['positiveFactors'][:2]))
        print("Negative factors:", ", ".join(prediction['negativeFactors'][:2]))
    
    # Test with low popularity songs
    low_popularity = df.nsmallest(5, 'popularity')
    print("\nLow Popularity Songs:")
    for _, song in low_popularity.iterrows():
        song_df = pd.DataFrame([song])
        prediction = predictor.predict(song_df)
        print(f"\nSong: {song['song']} by {song['artist']}")
        print(f"Year: {song['year']}")
        print(f"Actual popularity: {song['popularity']}")
        print(f"Predicted score: {prediction['score']:.2f}")
        print(f"Is a hit: {prediction['isHit']}")
        print("Positive factors:", ", ".join(prediction['positiveFactors'][:2]))
        print("Negative factors:", ", ".join(prediction['negativeFactors'][:2]))
    
    # Find and display the most correctly and incorrectly predicted songs
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
    
    # Convert to DataFrame for easier analysis
    predictions_df = pd.DataFrame(all_predictions)
    
    # Most accurate predictions
    print("\nMost Accurately Predicted Songs:")
    for _, pred in predictions_df.nsmallest(5, 'error').iterrows():
        print(f"{pred['song']} by {pred['artist']}: Actual={pred['actual']}, Predicted={pred['predicted']:.1f}")
    
    # Least accurate predictions
    print("\nLeast Accurately Predicted Songs:")
    for _, pred in predictions_df.nlargest(5, 'error').iterrows():
        print(f"{pred['song']} by {pred['artist']}: Actual={pred['actual']}, Predicted={pred['predicted']:.1f}")
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()