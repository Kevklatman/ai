import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

# Set absolute paths
BASE_DIR = '/Users/kevinklatman/Development/Code/ai'
DATA_DIR = os.path.join(BASE_DIR, 'data')

def create_historical_features(df, window_sizes=[3, 5, 10]):
    """
    Create historical features for each player based on their previous tournament performances.
    Uses pandas rolling windows for efficiency.
    """
    print("Processing data...")
    
    # Clean up column names and convert date
    df.columns = df.columns.str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert position to numeric, handling 'CUT' and other non-numeric values
    df['pos'] = pd.to_numeric(df['pos'].replace('CUT', 999), errors='coerce')
    
    # Create target variable
    df['is_top_10'] = (df['pos'] <= 10).astype(int)
    
    # Sort by player and date
    df = df.sort_values(['player_id', 'date'])
    
    # List of metrics to calculate rolling stats for
    metrics = ['sg_total', 'sg_t2g', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'made_cut']
    
    # Initialize progress bar
    pbar = tqdm(total=len(window_sizes) * len(metrics))
    
    # Calculate rolling stats for each window size
    for window in window_sizes:
        # Group by player and calculate rolling means
        for metric in metrics:
            col_name = f'{metric}_last_{window}'
            df[col_name] = df.groupby('player_id')[metric].transform(
                lambda x: x.shift().rolling(window=window, min_periods=1).mean()
            )
            pbar.update(1)
    
    pbar.close()
    
    # Keep only rows where we have some historical data
    df = df[df['date'] > df.groupby('player_id')['date'].transform('min')]
    
    # Select columns for final dataset
    feature_cols = [col for col in df.columns if any(f'_last_' in col for f in metrics)]
    keep_cols = ['player_id', 'player', 'tournament_id', 'tournament_name', 'date', 
                 'course', 'purse', 'season', 'is_top_10'] + feature_cols
    
    return df[keep_cols]

def main():
    try:
        # Load the raw dataset
        input_file = os.path.join(BASE_DIR, 'ASA All PGA Raw Data - Tourn Level.csv')
        output_file = os.path.join(DATA_DIR, 'pga_tour_historical.csv')
        
        print(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Create historical features
        historical_df = create_historical_features(df)
        
        # Save the processed dataset
        historical_df.to_csv(output_file, index=False)
        print(f"\nSaved historical features to: {output_file}")
        print(f"Processed data shape: {historical_df.shape}")
        
        # Print feature information
        print("\nHistorical features created:")
        feature_cols = [col for col in historical_df.columns 
                       if col not in ['player_id', 'player', 'tournament_id', 
                                    'tournament_name', 'date', 'course', 'purse', 'season']]
        for col in feature_cols:
            print(f"- {col}")
        
        # Print correlations with target
        correlations = historical_df[feature_cols].corr()['is_top_10'].sort_values(ascending=False)
        print("\nTop correlations with top 10 finish:")
        print(correlations.head(10))
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
