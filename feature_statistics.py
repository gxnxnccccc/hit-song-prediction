import pandas as pd
import numpy as np
import os

def analyze_audio_features(csv_file):
    """
    Analyze basic statistics of audio features from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing song data
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return
    
    # Load the dataset
    print(f"Loading dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded dataset with {len(df)} songs\n")
    
    # List of audio features to analyze
    audio_features = [
        'tempo', 'loudness', 'valence', 'energy', 
        'danceability', 'acousticness', 'instrumentalness', 
        'liveness', 'speechiness'
    ]
    
    # Check which features exist in the dataset
    missing_features = [f for f in audio_features if f not in df.columns]
    if missing_features:
        print(f"Warning: The following features are missing from the dataset: {missing_features}")
        audio_features = [f for f in audio_features if f in df.columns]
    
    # Create a DataFrame to store the statistics
    stats = pd.DataFrame(index=audio_features)
    
    # Calculate statistics for each feature
    stats['min'] = df[audio_features].min()
    stats['max'] = df[audio_features].max()
    stats['mean'] = df[audio_features].mean()
    stats['median'] = df[audio_features].median()
    stats['std'] = df[audio_features].std()
    
    # Add quartiles for more insight
    stats['25%'] = df[audio_features].quantile(0.25)
    stats['75%'] = df[audio_features].quantile(0.75)
    
    # Round to 4 decimal places for better readability
    stats = stats.round(4)
    
    # Print the statistics
    print("Audio Feature Statistics:")
    print("=" * 80)
    print(stats)
    print("=" * 80)
    
    # Calculate correlations with popularity (if available)
    if 'popularity' in df.columns:
        print("\nCorrelation with popularity:")
        corr_with_popularity = df[audio_features + ['popularity']].corr()['popularity'].drop('popularity').sort_values(ascending=False)
        for feature, corr in corr_with_popularity.items():
            print(f"{feature:20s}: {corr:.4f}")
    
    # Option to save statistics to a CSV file
    save_option = input("\nDo you want to save these statistics to a CSV file? (y/n): ").strip().lower()
    if save_option == 'y':
        output_file = "audio_feature_statistics.csv"
        stats.to_csv(output_file)
        print(f"Statistics saved to {output_file}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Prompt user for the CSV file
    csv_file = input("songs_filtered.csv").strip()
    
    # Run the analysis
    analyze_audio_features(csv_file)