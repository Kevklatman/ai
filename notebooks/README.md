# Notebooks Directory

This directory contains Jupyter notebooks for interactive data analysis and model development.

## Purpose
- Exploratory Data Analysis (EDA)
- Feature Engineering experiments
- Model training and evaluation
- Results visualization

## Contents

### 1. Initial Data Analysis (1_initial_data_analysis.ipynb)
#### Data Loading and Cleaning
- Loaded PGA Tour dataset (2015-2022)
- Handled missing values:
  - Dropped empty columns (Unnamed: 2,3,4)
  - Filled missing positions with 999
  - Filled missing Strokes Gained metrics with median values
  - Filled missing Finish values with 'Unknown'

#### Target Variable Creation
- Created binary 'is_top_10' variable (1 if position ≤ 10, 0 otherwise)

#### Correlation Analysis
Strong Predictors (|correlation| > 0.5):
- Fantasy points finish metrics (0.83-0.89)
- Total fantasy points (0.50-0.54)

Moderate Predictors (0.3 < |correlation| < 0.5):
- Streak and hole metrics (0.37-0.45)
- sg_total (0.41)
- sg_t2g (0.33)
- pos (-0.31)

Weak Predictors (|correlation| < 0.3):
- Individual strokes gained metrics
- Tournament characteristics
- Player characteristics
- Temporal features

### 2. Data Preprocessing (2_data_preprocessing.ipynb)
#### Feature Processing
- Load cleaned dataset
- Split into training and testing sets (80/20)
- Scale features using StandardScaler
- Save processed datasets for modeling

### Next Steps
1. Feature selection based on correlation analysis
2. Feature engineering
3. Model development and evaluation

## Usage
1. Start Jupyter notebook server
2. Open notebooks in browser
3. Run cells sequentially
4. Experiment with different approaches
