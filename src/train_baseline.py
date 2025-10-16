import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def train_baseline_model(data_path="data/processed/mental_health_clean.parquet"):
    """Train TF-IDF + Logistic Regression baseline model"""
    
    print("="*60)
    print("TRAINING BASELINE MODEL (TF-IDF + Logistic Regression)")
    print("="*60)
    
    # Load data
    df = pd.read_parquet(data_path)
    print(f"\nLoaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], 
        df['label'],
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga', random_state=42))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'tfidf__max_features': [5000, 10000, 15000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.01, 0.1, 1.0, 10.0]
    }
    
    # Grid search with cross-validation
    print("\n" + "="*60)
    print("Running Grid Search with 5-Fold Stratified CV...")
    print("="*60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe, 
        param_grid, 
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    
    gs.fit(X_train, y_train)
    
    print("\n" + "="*60)
    print("BEST PARAMETERS")
    print("="*60)
    print(gs.best_params_)
    print(f"\nBest CV F1 Score: {gs.best_score_:.4f}")
    
    # Get best model
    best_model = gs.best_estimator_
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Depression']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Depression'],
                yticklabels=['Normal', 'Depression'])
    plt.title('Confusion Matrix - Baseline Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/confusion_matrix_baseline.png', dpi=300)
    print("\nSaved confusion matrix: outputs/figures/confusion_matrix_baseline.png")
    
    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nPR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/pr_curve_baseline.png', dpi=300)
    print("Saved PR curve: outputs/figures/pr_curve_baseline.png")
    
    # Find optimal threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    
    print(f"\nOptimal threshold for F1: {best_threshold:.4f}")
    print(f"F1 at optimal threshold: {f1_scores[best_threshold_idx]:.4f}")
    
    # Top features
    print("\n" + "="*60)
    print("TOP FEATURES FOR DEPRESSION")
    print("="*60)
    
    feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
    coefficients = best_model.named_steps['clf'].coef_[0]
    top_indices = coefficients.argsort()[-30:][::-1]
    
    print("\nTop 30 features indicating depression:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:25s} (coef: {coefficients[idx]:.4f})")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/tfidf_lr_baseline.joblib'
    joblib.dump(best_model, model_path)
    print(f"\n" + "="*60)
    print(f"Model saved to: {model_path}")
    print("="*60)
    
    # Save metadata
    metadata = {
        'best_params': gs.best_params_,
        'cv_f1_score': gs.best_score_,
        'test_pr_auc': pr_auc,
        'test_roc_auc': roc_auc,
        'optimal_threshold': best_threshold,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    joblib.dump(metadata, 'models/baseline_metadata.joblib')
    print(f"Metadata saved to: models/baseline_metadata.joblib")
    
    return best_model, metadata

if __name__ == "__main__":
    if os.path.exists("data/processed/mental_health_clean.parquet"):
        model, metadata = train_baseline_model()
    else:
        print("Error: Processed data not found!")
        print("Please run src/preprocess.py first")