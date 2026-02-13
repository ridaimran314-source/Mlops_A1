"""
Data preprocessing: handle missing values, encode categoricals
"""
import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_data(input_file, output_file):
    """Preprocess the Titanic training data"""
    
    
    df = pd.read_csv(input_file)
    print(f"✓ Loaded raw data: {df.shape}")
    
    # Create a copy
    processed_df = df.copy()
    
    # Handle missing values
    # Age: fill with median
    processed_df['age'].fillna(processed_df['age'].median(), inplace=True)
    
    # Embarked: fill with mode
    if 'embarked' in processed_df.columns:
        processed_df['embarked'].fillna(processed_df['embarked'].mode()[0], inplace=True)
    
    # Fare: fill with median
    if 'fare' in processed_df.columns:
        processed_df['fare'].fillna(processed_df['fare'].median(), inplace=True)
    
    # Drop columns with too many missing values or not useful
    columns_to_drop = ['deck', 'embark_town', 'alive']
    processed_df.drop(columns=[col for col in columns_to_drop if col in processed_df.columns], 
                      inplace=True, errors='ignore')
    
    # Encode categorical variables
    # Sex: male=1, female=0
    if 'sex' in processed_df.columns:
        processed_df['sex'] = processed_df['sex'].map({'male': 1, 'female': 0})
    
    # Embarked: one-hot encoding
    if 'embarked' in processed_df.columns:
        embarked_dummies = pd.get_dummies(processed_df['embarked'], prefix='embarked', drop_first=True)
        processed_df = pd.concat([processed_df, embarked_dummies], axis=1)
        processed_df.drop('embarked', axis=1, inplace=True)
    
    # Who: one-hot encoding (if exists)
    if 'who' in processed_df.columns:
        who_dummies = pd.get_dummies(processed_df['who'], prefix='who', drop_first=True)
        processed_df = pd.concat([processed_df, who_dummies], axis=1)
        processed_df.drop('who', axis=1, inplace=True)
    
    
    # Class: one-hot encoding
    if 'class' in processed_df.columns:
        class_dummies = pd.get_dummies(processed_df['class'], prefix='class', drop_first=True)
        processed_df = pd.concat([processed_df, class_dummies], axis=1)
        processed_df.drop('class', axis=1, inplace=True)
    
    
    # Drop any remaining non-numeric columns except 'survived'
    non_numeric_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    processed_df.drop(columns=non_numeric_cols, inplace=True, errors='ignore')
    
    # Save processed data
    output_path = Path(output_file)
    processed_df.to_csv(output_path, index=False)
    
    print(f"✓ Preprocessing complete")
    print(f"  - Output shape: {processed_df.shape}")
    print(f"  - Saved to: {output_path}")
    print(f"  - Features: {processed_df.columns.tolist()}")
    

if __name__ == "__main__":
    """Preprocess both train and test datasets"""
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Preprocessing datasets...")
    
    # Process training data
    train_input = input_dir / "train.csv"
    train_output = output_dir / "train_processed.csv"
    preprocess_data(train_input, train_output)
    
    # Process test data
    test_input = input_dir / "test.csv"
    test_output = output_dir / "test_processed.csv"
    preprocess_data(test_input, test_output)