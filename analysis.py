import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score
from data_loader import load_data, load_test_data
from model import ModelTrainer
from preprocessor import DataPreprocessor
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec


def analyze_performance(X, y, model, preprocessor):
    X_scaled = preprocessor.scale_data(X, 'standard')
    cv_scores = cross_val_score(model, X_scaled, y, cv=4)
    y_pred = cross_val_predict(model, X_scaled, y, cv=4)

    print("\nCross-validation scores:", cv_scores)
    print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    return y_pred, X_scaled


def create_visualizations(y_true, y_pred, probas, classes):
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2)

    # Confusion Matrix
    plt.subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Confidence Distribution
    plt.subplot(gs[0, 1])
    sns.histplot(np.max(probas, axis=1), bins=50)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')

    # Per-class Performance
    plt.subplot(gs[1, 0])
    class_accuracies = []
    for cls in classes:
        mask = y_true == cls
        acc = np.mean(y_pred[mask] == y_true[mask])
        class_accuracies.append(acc)
    plt.bar(classes, class_accuracies)
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)

    # Error Analysis
    plt.subplot(gs[1, 1])
    errors = y_true != y_pred
    error_confidence = np.max(probas[errors], axis=1)
    sns.boxplot(x=y_true[errors], y=error_confidence)
    plt.title('Error Confidence by Class')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    return cm, class_accuracies


def analyze_misclassifications(X, y, y_pred, model, X_scaled):
    probas = model.predict_proba(X_scaled)
    misclassified = y != y_pred
    misclassified_indices = np.where(misclassified)[0]

    error_analysis = {
        'total_errors': len(misclassified_indices),
        'confident_errors': len(np.where((np.max(probas, axis=1) > 0.8) & misclassified)[0]),
        'probas': probas
    }

    print(f"\nTotal misclassified: {error_analysis['total_errors']}")
    print(f"Confident errors: {error_analysis['confident_errors']}")

    return error_analysis


def visualize_feature_space(X_scaled, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    classes = ['person', 'sign', 'bike', 'bus', 'car']
    for cls in classes:
        mask = y == cls
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=cls, alpha=0.6)

    plt.title('t-SNE Feature Space Visualization')
    plt.legend()
    plt.savefig('feature_space.png', dpi=300, bbox_inches='tight')
    plt.close()


def process_test_data(X_test, model, preprocessor):
    X_test_scaled = preprocessor.scale_data(X_test)
    predictions = model.predict(X_test_scaled)
    data_mapping = {'person': 0, 'sign': 1, 'bike': 2, 'bus': 3, 'car': 4}
    mapped_predictions = [data_mapping[pred] for pred in predictions]

    submission = pd.DataFrame({'prediction': mapped_predictions})
    submission.to_csv('submission.csv', index=False)

    return predictions, model.predict_proba(X_test_scaled)


def main():
    print("Loading data...")
    X, y = load_data()
    X_test = load_test_data()

    preprocessor = DataPreprocessor()
    model = ModelTrainer().classifiers['svm']

    print("\nAnalyzing performance...")
    y_pred, X_scaled = analyze_performance(X, y, model, preprocessor)

    print("\nGenerating visualizations...")
    model.fit(X_scaled, y)
    probas = model.predict_proba(X_scaled)
    classes = ['person', 'sign', 'bike', 'bus', 'car']
    cm, class_accuracies = create_visualizations(y, y_pred, probas, classes)

    print("\nAnalyzing errors...")
    error_analysis = analyze_misclassifications(X, y, y_pred, model, X_scaled)

    print("\nVisualizing feature space...")
    visualize_feature_space(X_scaled, y)

    print("\nProcessing test data...")
    test_predictions, test_probas = process_test_data(X_test, model, preprocessor)

    return {
        'confusion_matrix': cm,
        'class_accuracies': dict(zip(classes, class_accuracies)),
        'error_analysis': error_analysis,
        'test_predictions': test_predictions,
        'test_probabilities': test_probas
    }


if __name__ == "__main__":
    results = main()