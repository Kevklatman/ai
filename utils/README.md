# Utils Directory

This directory contains helper functions and utilities used across the project.

## Current Contents

### data_preprocessing.py
A utility module for preprocessing PGA Tour data with the following key functions:

#### 1. `load_and_clean_data(file_path: str) -> pd.DataFrame`
- **Purpose**: Loads and performs initial cleaning of the PGA Tour dataset
- **Input**: Path to the CSV file
- **Output**: Cleaned pandas DataFrame
- **Key Operations**:
  - Loads the dataset
  - Creates target variable (top 10 finish)

#### 2. `prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]`
- **Purpose**: Prepares feature set for modeling
- **Input**: Input DataFrame
- **Output**: Processed DataFrame and list of selected features
- **Key Features**:
  - made_cut
  - finish_strokes
  - total_strokes
  - total_rounds

#### 3. `split_data(df: pd.DataFrame, features: List[str], target: str, test_size: float, random_state: int)`
- **Purpose**: Splits data into training and testing sets
- **Input**: 
  - DataFrame
  - List of features
  - Target variable name
  - Test size proportion
  - Random state
- **Output**: X_train, X_test, y_train, y_test arrays
- **Default Parameters**:
  - test_size: 0.2 (20% test data)
  - random_state: 42 (for reproducibility)

## Future Additions
The following utilities will be added as needed:
- feature_utils.py: Feature engineering helper functions
- evaluation_utils.py: Model evaluation utilities
- visualization.py: Common plotting functions

## Usage Example
```python
from utils.data_preprocessing import load_and_clean_data, prepare_features, split_data

# Load and clean data
df = load_and_clean_data('path/to/pga_data.csv')

# Prepare features
X, features = prepare_features(df)

# Split data
X_train, X_test, y_train, y_test = split_data(df, features)
```

## Dependencies
- pandas
- numpy
- scikit-learn

## Notes
- All functions include type hints for better code documentation
- Functions are designed to be modular and reusable
- Additional preprocessing steps will be added as needed
