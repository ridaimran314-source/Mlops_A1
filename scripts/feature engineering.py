import pandas as pd
import numpy as np
from pathlib import Path

def engineer_features(input_file, output_file):
    
    df = pd.read_csv(input_file)
    
    # Create new features
    feature_df = df.copy()
    
    # Family size feature
    if 'sibsp' in feature_df.columns and 'parch' in feature_df.columns:
        feature_df['family_size'] = feature_df['sibsp'] + feature_df['parch'] + 1
    
    # Is alone feature (if not already exists)
    if 'family_size' in feature_df.columns and 'alone' not in feature_df.columns:
        feature_df['is_alone'] = (feature_df['family_size'] == 1).astype(int)
    
    # Age bins
    if 'age' in feature_df.columns:
        feature_df['age_bin'] = pd.cut(feature_df['age'], 
                                        bins=[0, 12, 18, 35, 60], 
                                        labels=[0, 1, 2, 3]).astype(int)
    
    # Fare bins
    if 'fare' in feature_df.columns:
        feature_df['fare_bin'] = pd.qcut(feature_df['fare'], 
                                          q=4, 
                                          labels=[0, 1, 2, 3], 
                                          duplicates='drop').astype(int)
    
    # Fare per person
    feature_df['fare_per_person'] = feature_df['fare'] / feature_df['family_size']
    
    # Select relevant features for modeling
    # Keep survived as target
    target_col = 'survived'
    
    # Feature columns (exclude target)
    feature_columns = [col for col in feature_df.columns if col != target_col]
    
    # Save engineered features
    feature_df.to_csv(output_file, index=False)
    
    print(f"✓ Feature engineering complete")
    print(f"  - Output shape: {feature_df.shape}")
    print(f"  - Total features: {len(feature_columns)}")
    print(f"  - Saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    input_dir = Path("data/processed")
    output_dir = Path("features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Engineering features...")
    
    # Process training data
    train_input = input_dir / "train_processed.csv"
    train_output = output_dir / "train_features.csv"
    train_df = engineer_features(train_input, train_output)
    
    # Process test data
    test_input = input_dir / "test_processed.csv"
    test_output = output_dir / "test_features.csv"
    test_df = engineer_features(test_input, test_output)