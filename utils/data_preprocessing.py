"""
Utility functions for preprocessing PGA Tour data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load the PGA Tour dataset and perform initial cleaning.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Create target variable
    df['is_top_10'] = df['pos'] <= 10
    
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for modeling.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Processed dataframe and list of selected features
    """
    # Select initial features (to be expanded)
    selected_features = [
        'made_cut',
        'finish_strokes',
        'total_strokes',
        'total_rounds'
    ]
    
    # Ensure all features exist
    available_features = [f for f in selected_features if f in df.columns]
    
    return df[available_features], available_features

def split_data(df: pd.DataFrame, 
               features: List[str], 
               target: str = 'is_top_10',
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataset
        features (List[str]): List of feature columns
        target (str): Target variable name
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df[features]
    y = df[target]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
