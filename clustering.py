from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

def kmeans_clustering(data, n_clusters=3):
    """Performs K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

def hierarchical_clustering(data, method="ward"):
    """Performs hierarchical clustering."""
    clustering = AgglomerativeClustering(linkage=method)
    labels = clustering.fit_predict(data)
    return labels
