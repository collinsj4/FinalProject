from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE


class DataPreprocessor:
    def __init__(self):
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }

        # Initialize PCA configurations
        self.pca_configs = {
            'pca_50': PCA(n_components=50),
            'pca_100': PCA(n_components=100)
        }

        # Initialize clustering methods
        self.clustering = {
            'kmeans': KMeans(n_clusters=5, random_state=42),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }

    def scale_data(self, X, scaler_type='standard'):
        """Apply scaling to the data"""
        try:
            return self.scalers[scaler_type].fit_transform(X)
        except Exception as e:
            print(f"Error in scaling data: {e}")
            raise

    def reduce_dimensions(self, X, method='pca_50'):
        """Apply dimensionality reduction"""
        try:
            return self.pca_configs[method].fit_transform(X)
        except Exception as e:
            print(f"Error in dimensionality reduction: {e}")
            raise

    def cluster_data(self, X, method='kmeans'):
        """Apply clustering to the data"""
        try:
            return self.clustering[method].fit_predict(X)
        except Exception as e:
            print(f"Error in clustering: {e}")
            raise

    def create_feature_selectors(self, estimator):
        """Create feature selection methods"""
        return {
            'mutual_info': SelectKBest(mutual_info_classif, k=100),
            'rfe': RFE(estimator=estimator, n_features_to_select=100)
        }