import pandas as pd
import numpy as np
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    average_precision_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_transformer_model(data_path="data/processed/mental_health_clean.parquet",
                           model_name="distilbert-base-uncased",
                           output_dir="models/distilbert_finetuned"):
    """Fine-tune DistilBERT for depression detection"""
    
    print("="*60)
    print(f"TRAINING TRANSFORMER MODEL: {model_name}")
    print("="*60)
    
    # Load data
    df = pd.read_parquet(data_path)
    print(f"\nLoaded {len(df)} samples")
    
    # Prepare dataset
    df_model = df[['text', 'label']].copy()
    ds = Dataset.from_pandas(df_model)
    
    # Train-test split
    ds = ds.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
    print(f"\nTrain set: {len(ds['train'])} samples")
    print(f"Test set: {len(ds['test'])} samples")
    
    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=256
        )
    
    print("Tokenizing dataset...")
    tokenized_ds = ds.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        seed=42
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    train_result = trainer.train()
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    predictions = trainer.predict(tokenized_ds['test'])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    y_proba = np.exp(predictions.predictions) / np.exp(predictions.predictions).sum(axis=-1, keepdims=True)
    y_proba_pos = y_proba[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Depression']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'Depression'],
                yticklabels=['Normal', 'Depression'])
    plt.title('Confusion Matrix - Transformer Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/confusion_matrix_transformer.png', dpi=300)
    print("\nSaved confusion matrix: outputs/figures/confusion_matrix_transformer.png")
    
    # Calculate additional metrics
    pr_auc = average_precision_score(y_true, y_proba_pos)
    roc_auc = roc_auc_score(y_true, y_proba_pos)
    
    print(f"\nPR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Save model and tokenizer
    print("\n" + "="*60)
    print("Saving model and tokenizer...")
    print("="*60)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")
    
    # Save metadata
    import json
    metadata = {
        'model_name': model_name,
        'train_samples': len(ds['train']),
        'test_samples': len(ds['test']),
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'training_args': training_args.to_dict()
    }
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {output_dir}/metadata.json")
    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return trainer, metadata

if __name__ == "__main__":
    if os.path.exists("data/processed/mental_health_clean.parquet"):
        # Check if CUDA is available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")
        
        if device == "cpu":
            print("\nWARNING: Training on CPU will be slow!")
            print("Consider using Google Colab with GPU for faster training.")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Training cancelled.")
                exit()
        
        trainer, metadata = train_transformer_model()
    else:
        print("Error: Processed data not found!")
        print("Please run src/preprocess.py first")