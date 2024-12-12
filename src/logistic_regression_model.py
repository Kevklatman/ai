import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
BASE_DIR = '/Users/kevinklatman/Development/Code/ai'
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load the historical features dataset and prepare it for modeling."""
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'pga_tour_historical.csv'))
    
    # Get feature columns (excluding metadata and target)
    feature_cols = [col for col in df.columns if '_last_' in col]
    
    # Prepare features and target
    X = df[feature_cols]
    y = df['is_top_10']
    
    # Store tournament info for later analysis
    tournament_info = df[['tournament_id', 'player', 'date', 'is_top_10']]
    
    # Print class distribution
    print("\nClass distribution in full dataset:")
    print(y.value_counts(normalize=True))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Also split tournament info
    _, tournament_info_test = train_test_split(
        tournament_info, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, tournament_info_test

def analyze_predictions_by_tournament(tournament_info, y_pred, y_pred_proba, model, feature_cols):
    """Analyze predictions and show which stats contributed to high probabilities."""
    # Combine predictions with tournament info
    predictions_df = pd.DataFrame({
        'tournament_id': tournament_info['tournament_id'],
        'player': tournament_info['player'],
        'date': pd.to_datetime(tournament_info['date']),
        'actual_top_10': tournament_info['is_top_10'],
        'predicted_probability': y_pred_proba
    })
    
    # Add threshold flags
    thresholds = [0.7, 0.6, 0.5, 0.4, 0.3]
    for threshold in thresholds:
        predictions_df[f'predicted_{threshold}'] = (y_pred_proba >= threshold).astype(int)
    
    # Group by tournament and analyze predictions at each threshold
    print("\nPrediction Analysis by Threshold:")
    for threshold in thresholds:
        n_predicted = predictions_df[f'predicted_{threshold}'].sum()
        n_correct = ((predictions_df[f'predicted_{threshold}'] == 1) & 
                    (predictions_df['actual_top_10'] == 1)).sum()
        print(f"\nThreshold {threshold}:")
        print(f"Total players predicted: {n_predicted}")
        print(f"Correct predictions: {n_correct}")
        print(f"Success rate: {n_correct/n_predicted*100:.1f}%")
    
    # Analyze predictions per tournament
    print("\nDetailed Tournament Analysis:")
    tournament_predictions = predictions_df.groupby('tournament_id').agg({
        'player': 'count',          # Number of players in tournament
        'actual_top_10': 'sum',     # Number of actual top 10s (should be 10)
        'date': 'first'             # Tournament date
    }).sort_values('date')
    
    # Add prediction counts for each threshold
    for threshold in thresholds:
        col_name = f'predicted_{threshold}'
        tournament_predictions[col_name] = \
            predictions_df.groupby('tournament_id')[f'predicted_{threshold}'].sum()
    
    print("\nSample of tournament predictions:")
    print(tournament_predictions.head().to_string())
    
    # Show distribution of number of predictions
    print("\nFor each threshold, number of tournaments with X predicted players:")
    for threshold in thresholds:
        col_name = f'predicted_{threshold}'
        prediction_counts = tournament_predictions[col_name].value_counts().sort_index()
        print(f"\nThreshold {threshold}:")
        print("Num Players Predicted    Num Tournaments")
        print("-" * 45)
        for n_players, n_tournaments in prediction_counts.items():
            print(f"{n_players:>20d}    {n_tournaments:>15d}")
    
    # Calculate averages
    print("\nAverage predictions per tournament:")
    for threshold in thresholds:
        avg_predictions = tournament_predictions[f'predicted_{threshold}'].mean()
        print(f"Threshold {threshold}: {avg_predictions:.1f} players")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 most important features:")
    print(feature_importance.head().to_string())
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    for threshold in thresholds:
        counts = tournament_predictions[f'predicted_{threshold}'].value_counts().sort_index()
        plt.plot(counts.index, counts.values, 
                label=f'Threshold {threshold}', 
                alpha=0.7, 
                marker='o')
    
    plt.axvline(x=10, color='r', linestyle='--', label='Actual Top 10')
    plt.xlabel('Number of Players Predicted per Tournament')
    plt.ylabel('Number of Tournaments')
    plt.title('How Many Tournaments Had X Players Predicted\nAt Different Thresholds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'predictions_per_tournament.png'))
    plt.close()
    
    return predictions_df

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train logistic regression model and return predictions."""
    # Calculate class weights
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    class_weights = dict(zip(
        np.unique(y_train),
        n_samples / (n_classes * np.bincount(y_train))
    ))
    
    # Train model with class weights
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight=class_weights,
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    
    # Make predictions with different thresholds
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}
    
    print("\nPerformance at different thresholds:")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        results[threshold] = y_pred
        print(f"\nThreshold: {threshold}")
        print(classification_report(y_test, y_pred))
    
    # Use 0.5 as default threshold for final predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return model, y_pred, y_pred_proba, results

def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance)), importance['importance'])
    plt.yticks(range(len(importance)), importance['feature'])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance in Predicting Top 10 Finish')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return importance

def plot_confusion_matrix(y_test, y_pred, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_probability_distribution(y_test, y_pred_proba, output_path):
    """Plot probability distribution for actual top 10 vs non-top 10."""
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Non-Top 10', density=True)
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Top 10', density=True)
    plt.xlabel('Predicted Probability of Top 10 Finish')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, output_path):
    """Plot ROC curve and calculate AUC."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Print some threshold analysis
    print("\nThreshold Analysis:")
    for i in range(len(thresholds)):
        if i % (len(thresholds) // 5) == 0:  # Print ~5 evenly spaced thresholds
            print(f"Threshold {thresholds[i]:.2f}: TPR = {tpr[i]:.2f}, FPR = {fpr[i]:.2f}")
    
    return roc_auc

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_cols, tournament_info_test = load_and_prepare_data()
    
    # Train model and get predictions
    model, y_pred, y_pred_proba, threshold_results = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Analyze predictions by tournament
    predictions_df = analyze_predictions_by_tournament(tournament_info_test, y_pred, y_pred_proba, model, feature_cols)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    importance = plot_feature_importance(
        model, feature_cols,
        os.path.join(FIGURES_DIR, 'feature_importance.png')
    )
    print("\nTop Features by Importance:")
    print(importance.tail(10).to_string())
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    )
    
    # Plot probability distribution
    plot_probability_distribution(
        y_test, y_pred_proba,
        os.path.join(FIGURES_DIR, 'probability_distribution.png')
    )
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(
        y_test, y_pred_proba,
        os.path.join(FIGURES_DIR, 'roc_curve.png')
    )

if __name__ == "__main__":
    main()
