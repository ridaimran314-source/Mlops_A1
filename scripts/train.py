import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def train_model(model_type='random_forest'):
    """Train the classification model"""
    # Load features
    input_path = Path("features/train_features.csv")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Split data:")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Validation: {X_val.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        print("✓ Training Random Forest model...")
    else:  # SVM
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True
        )
        print("✓ Training SVM model...")
    
    model.fit(X_train_scaled, y_train)
    
    # Validation score
    val_score = model.score(X_val_scaled, y_val)
    train_score = model.score(X_train_scaled, y_train)
    
    print(f"✓ Model trained successfully")
    print(f"  - Train accuracy: {train_score:.4f}")
    print(f"  - Validation accuracy: {val_score:.4f}")
    
    # Save model and scaler
    model_path = models_dir / "model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")
    
    return True

if __name__ == "__main__":
    success = train_model('random_forest')