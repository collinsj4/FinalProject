import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def merge_datasets(main_file, extra_file):
    """Merges the main dataset with extra hard samples."""
    main_data = pd.read_csv(main_file)
    extra_data = pd.read_csv(extra_file)
    combined_data = pd.concat([main_data, extra_data], axis=0).reset_index(drop=True)
    return combined_data

def normalize_data(data, method="z-score"):
    """Normalizes the data."""
    scaler = StandardScaler() if method == "z-score" else MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns)

def split_data(data, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    X = data.drop('class', axis=1)
    y = data['class']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
