import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data():
    # Set up absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load the cleaned dataset
    df = pd.read_csv(os.path.join(data_dir, 'pga_tour_cleaned.csv'))
    
    # Print column names to check what's available
    print("Available columns:")
    print(df.columns.tolist())
    
    # Select only legitimate predictive features
    features = [
        'sg_total',      # Total strokes gained
        'sg_t2g',        # Strokes gained tee to green
        'n_rounds',      # Number of rounds played
        'made_cut',      # Whether player made the cut
        'hole_par',      # Par for the hole
        'strokes'        # Total strokes
    ]
    
    # Prepare features (X) and target (y)
    X = df[features]
    y = df['is_top_10']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame to preserve feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
    
    # Save processed datasets
    X_train_scaled.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    print("\nData preprocessing completed. Files saved in data/processed/")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
