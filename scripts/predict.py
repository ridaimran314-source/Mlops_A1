import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def generate_predictions():
    """Generate predictions for test/validation data"""
    # Load model and scaler
    models_dir = Path("models")
    model = joblib.load(models_dir / "model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    print("✓ Loaded model and scaler")
    
    # Load features (using part of training data as "test")
    features_path = Path("features/test_features.csv")
    df = pd.read_csv(features_path)
    
    # Use last 20% as "test set" for prediction
    test_size = int(0.2 * len(df))
    test_df = df.tail(test_size).copy()
    
    # Separate features and true labels
    X_test = test_df.drop('survived', axis=1)
    y_true = test_df['survived']
    
    print(f"✓ Loaded test data: {X_test.shape}")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Generate predictions
    predictions = model.predict(X_test_scaled) # 0 or 1
    probabilities = model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1 (survived) [0-1]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'true_label': y_true.values,
        'predicted_label': predictions,
        'probability': probabilities
    })
    
    # Save predictions
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.csv"
    
    results_df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions generated")
    print(f"  - Total predictions: {len(predictions)}")
    print(f"  - Positive class: {predictions.sum()}")
    print(f"  - Negative class: {len(predictions) - predictions.sum()}")
    print(f"  - Saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    success = generate_predictions()