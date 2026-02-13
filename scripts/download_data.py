
"""
Download Titanic dataset from Kaggle or OpenML
"""
import pandas as pd
import os
from pathlib import Path

def download_titanic_data():
    """Download Titanic dataset using sklearn or direct URL"""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try using seaborn's built-in dataset
        import seaborn as sns
        df = sns.load_dataset('titanic')

        # Split into train and test (similar to Kaggle competition)
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # Save datasets
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        print(f"✓ Downloaded Titanic dataset")
        print(f"  - Train: {len(train_df)} rows")
        print(f"  - Test: {len(test_df)} rows")
        print(f"  - Saved to: {output_dir}")

        return True

    except Exception as e:
        print(f"✗ Error downloading data: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_titanic_data()
    exit(0 if success else 1)