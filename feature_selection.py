import pandas as pd
import numpy as np

def select_features_variance(data, threshold=0.1):
    """Selects features based on variance threshold."""
    variances = data.var()
    selected_features = variances[variances > threshold].index
    return data[selected_features]

def select_features_correlation(data, target, threshold=0.3):
    """Selects features based on correlation with the target."""
    correlations = data.corrwith(target)
    selected_features = correlations[abs(correlations) > threshold].index
    return data[selected_features]
