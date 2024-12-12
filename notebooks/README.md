# Notebooks Directory

This directory contains Jupyter notebooks for the PGA Tour prediction project.

## Notebooks Overview

### 1. Data Analysis (`1_data_analysis.ipynb`)
- Initial data exploration
- Feature distributions and relationships
- Correlation analysis
- Target variable analysis

### 2. Data Preprocessing (`2_data_preprocessing.ipynb`)
- Data cleaning and feature selection
- Feature engineering
- Train-test split (80-20)
- Data scaling and normalization
- Creation of processed datasets in `data/processed/`

### 3. Logistic Regression Model (`3_model_logistic_regression.ipynb`)
- First model implementation using logistic regression
- Features:
  - Strokes gained metrics (total, tee-to-green)
  - Fantasy points (FDP, DKP, SDP)
  - Player statistics (made cuts, rounds played)
- Model evaluation:
  - Classification metrics
  - Confusion matrix
  - ROC curve analysis
  - Feature importance visualization

## Running the Notebooks
1. Start with `1_data_analysis.ipynb` for data understanding
2. Run `2_data_preprocessing.ipynb` to prepare the data
3. Execute `3_model_logistic_regression.ipynb` for model training and evaluation

## Dependencies
Required Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
