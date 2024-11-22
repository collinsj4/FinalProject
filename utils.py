import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def save_model(model, filename):
    """Saves a trained model."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """Loads a saved model."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def evaluate_model(predictions, ground_truth):
    """Prints classification report and confusion matrix."""
    print(classification_report(ground_truth, predictions))
    print("Confusion Matrix:\n", confusion_matrix(ground_truth, predictions))

def plot_results(data, labels):
    """Visualizes results."""
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.colorbar()
    plt.title("Results Visualization")
    plt.show()
