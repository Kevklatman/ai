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

### Next Immediate Steps
1. Feature Engineering
   - Create derived features from strong predictors
   - Remove weak predictors
   - Handle categorical variables

2. Data Preprocessing
   - Split data into training and testing sets
   - Scale features
   - Prepare for model training

## Project Roadmap

### Phase 2: Basic Model Implementation
1. **Data Preprocessing**
   - Split data into training and testing sets
   - Scale features
   - Handle categorical variables
   
2. **First Model: Logistic Regression**
   - Implement simple logistic regression
   - Train the model
   - Make predictions
   - Evaluate basic performance

### Phase 3: Model Evaluation and Improvement
1. **Performance Analysis**
   - Calculate accuracy, precision, recall
   - Generate classification report
   - Analyze feature importance
   - Identify model weaknesses

2. **Model Iteration**
   - Try different algorithms:
     - Random Forest
     - Support Vector Machine
     - Gradient Boosting
   - Compare performance
   - Select best performing model

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

## Implementation Details

### Tools and Libraries
- **Python**: Primary programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **numpy**: Numerical computations

### Directory Structure
```
ai/
├── README.md           # Project roadmap (this file)
├── initial_set/        # Initial data exploration
│   ├── README.md      # Data exploration documentation
│   └── python.py      # Data analysis script
├── data/              # Processed datasets
│   └── pga_tour_cleaned.csv  # Cleaned dataset with selected features
├── models/            # Different model implementations
├── notebooks/         # Jupyter notebooks for analysis
└── utils/            # Helper functions and utilities
```

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

## Getting Started
1. Clone the repository
2. Install required dependencies
3. Run initial data analysis
4. Follow along with each phase
5. Experiment and modify as needed

Remember: The goal is not just to build a working model, but to understand each step of the machine learning process. Take time to experiment and understand why each step is important.
