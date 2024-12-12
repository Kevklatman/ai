# Machine Learning Journey: PGA Tour Prediction Project 🏌️‍♂️🤖

## Project Goal
Build a machine learning model that predicts whether a PGA Tour player will finish in the top 10 of a tournament. This project serves as an introduction to machine learning concepts using real-world sports data.

## Learning Objectives
- Understand the basic machine learning workflow
- Learn data preprocessing techniques
- Implement a classification model
- Evaluate model performance
- Interpret results and improve the model

## Current Progress

### Phase 1: Data Exploration and Preparation ✅
1. **Initial Data Analysis** (`notebooks/1_initial_data_analysis.ipynb`)
   - ✅ Loaded and examined PGA Tour dataset (2015-2022)
   - ✅ Handled missing values:
     - Dropped empty columns (Unnamed: 2,3,4)
     - Filled missing positions with 999
     - Filled missing Strokes Gained metrics with median values
     - Filled missing Finish values with 'Unknown'
   - ✅ Created target variable (is_top_10)
   - ✅ Analyzed feature correlations

2. **Feature Selection Insights**
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

### Phase 2: Basic Model Implementation ✅
1. **Data Preprocessing**
   - ✅ Split data into training and testing sets
   - ✅ Scale features
   - ✅ Handle categorical variables
   
2. **First Model: Logistic Regression**
   - ✅ Implemented logistic regression as baseline model
   - ✅ Features used include: strokes gained metrics for last 3, 5, and 10 tournaments
   - ✅ Model evaluation metrics implemented:
     - Classification report
     - Prediction thresholds (0.3-0.7)
     - Feature importance analysis

### Phase 3: Database Implementation ✅
1. **SQLite Database Creation**
   - ✅ Created `pga_predictions.db` with tables for:
     - Tournaments
     - Players
     - Historical stats
     - Predictions
   - ✅ Added views for easy analysis:
     - Prediction accuracy by threshold
     - Player performance tracking

2. **Key Findings**
   - Model achieves 28.9% accuracy at 0.7 confidence threshold
   - Most important features:
     1. Strokes gained total (last 5 tournaments)
     2. Strokes gained off the tee (last 10 tournaments)
     3. Strokes gained tee to green (last 5 tournaments)
   - Top performing predictions for players like J.B. Holmes, Matt Kuchar, and Rickie Fowler

### Next Steps
1. **Model Improvement**
   - Experiment with different algorithms (Random Forest, XGBoost)
   - Feature engineering based on discovered insights
   - Implement cross-validation

2. **Analysis Enhancement**
   - Create visualization dashboard
   - Track prediction accuracy over time
   - Analyze course-specific performance

## Project Structure
```
.
├── data/
│   ├── pga_tour_historical.csv
│   └── pga_predictions.db (SQLite database)
├── notebooks/
│   └── 1_initial_data_analysis.ipynb
├── src/
│   ├── create_database.py
│   └── logistic_regression_model.py
├── figures/
│   └── predictions_per_tournament.png
├── README.md
└── requirements.txt
```

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the model: `python src/logistic_regression_model.py`
4. Create database: `python src/create_database.py`

## Dependencies
- pandas
- numpy
- scikit-learn
- sqlite3
- matplotlib
- seaborn

## Note
The SQLite database (*.db files) is excluded from version control to keep the repository size manageable. Run `create_database.py` to generate it locally.

## Implementation Details

### Tools and Libraries
- **Python**: Primary programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **numpy**: Numerical computations

### Success Metrics
1. **Model Performance**
   - Accuracy > 70%
   - Balanced precision and recall
   - Good performance on unseen data

2. **Learning Goals**
   - Understanding of ML workflow
   - Ability to preprocess data
   - Knowledge of model evaluation
   - Experience with real-world data

## Project Roadmap

### Phase 4: Advanced Techniques
1. **Feature Selection**
   - Implement feature importance analysis
   - Remove irrelevant features
   - Add new derived features

2. **Hyperparameter Tuning**
   - Use cross-validation
   - Implement grid search
   - Fine-tune model parameters

### Phase 5: Final Implementation
1. **Model Deployment**
   - Save trained model
   - Create prediction pipeline
   - Implement easy-to-use interface

2. **Documentation**
   - Document model performance
   - Create usage instructions
   - Note potential improvements

Remember: The goal is not just to build a working model, but to understand each step of the machine learning process. Take time to experiment and understand why each step is important.
