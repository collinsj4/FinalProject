import pandas as pd
import numpy as np


def load_data():
    """Load and combine both datasets"""
    try:
        # Load both datasets
        df1 = pd.read_csv('data/Data.csv')
        df2 = pd.read_csv('data/extra_hard_samples.csv')

        # Combine datasets
        df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

        # Split features and target
        X = df.iloc[:, 2:].values  # All features (excluding image_name and class)
        y = df['class'].values

        print(f"Loaded {len(df)} samples with {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise