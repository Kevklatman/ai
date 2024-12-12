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
    
    # Print class distribution
    print("\nClass distribution in full dataset:")
    print(y.value_counts(normalize=True))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

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
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.3  # Lower threshold to catch more potential top 10 finishes
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return model, y_pred, y_pred_proba

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
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
    
    # Train model and get predictions
    model, y_pred, y_pred_proba = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
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
