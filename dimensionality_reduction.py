import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca(data, n_components=2):
    """Applies PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def visualize_pca(data, labels):
    """Plots PCA-reduced data."""
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA Visualization')
    plt.colorbar()
    plt.show()
