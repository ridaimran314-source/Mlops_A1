import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json

def evaluate_model():
    """Evaluate model predictions"""
    # Load predictions
    predictions_path = Path("results/predictions.csv")
    results_df = pd.read_csv(predictions_path)
    
    print("✓ Loaded predictions")
    
    # Extract true labels and predictions
    y_true = results_df['true_label']
    y_pred = results_df['predicted_label']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print("✓ Evaluation Metrics:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"\n✓ Confusion Matrix:")
    print(f"  {cm}")
    
    # Save metrics to file
    output_dir = Path("results")
    metrics_path = output_dir / "metrics.txt"
    
    with open(metrics_path, 'w') as f:
        f.write("MODEL EVALUATION METRICS\n\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("=" * 50 + "\n")
        f.write("\nClassification Report:\n")
        f.write("=" * 50 + "\n")
        f.write(classification_report(y_true, y_pred))
    
    print(f"✓ Metrics saved to: {metrics_path}")
    
    return True

if __name__ == "__main__":
    success = evaluate_model()