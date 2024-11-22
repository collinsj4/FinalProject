from data_preprocessing import load_data, normalize_data, split_data
from dimensionality_reduction import apply_pca, visualize_pca
from clustering import kmeans_clustering
from feature_selection import select_features_variance
from classifiers import knn_classifier
from utils import evaluate_model
import pandas as pd


def main():
    # Step 1: Load and preprocess data
    print("Loading data...")
    data = load_data("data/Data.csv")

    # Load additional hard samples and concatenate
    print("Loading extra hard samples and combining datasets...")
    hard_samples = pd.read_csv("data/extra_hard_samples.csv")
    data = pd.concat([data, hard_samples], axis=0).reset_index(drop=True)

    # Normalize the data
    print("Normalizing data...")
    normalized_data = normalize_data(data)

    # Split data into train and test sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(normalized_data)

    # Step 2: Dimensionality reduction using PCA
    print("Applying PCA for dimensionality reduction...")
    reduced_data = apply_pca(X_train)
    print("Visualizing PCA results...")
    visualize_pca(reduced_data, y_train)

    # Step 3: Perform clustering
    print("Clustering the reduced data...")
    clusters = kmeans_clustering(reduced_data)
    print(f"Clustering completed. Number of clusters: {len(set(clusters))}")

    # Step 4: Feature selection
    print("Selecting important features...")
    selected_data = select_features_variance(X_train)
    print(f"Selected {selected_data.shape[1]} features based on variance.")

    # Step 5: Classification using KNN
    print("Training and testing the KNN classifier...")
    predictions = knn_classifier(X_train, y_train, X_test)

    # Step 6: Evaluate the model
    print("Evaluating model performance...")
    evaluate_model(predictions, y_test)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
