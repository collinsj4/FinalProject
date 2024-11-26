from data_loader import load_data
from preprocessor import DataPreprocessor
from model import ModelTrainer
from utils import print_results, find_best_method
import warnings

warnings.filterwarnings('ignore')


def run_phase1():
    """Phase 1: Testing different scalers"""
    X, y = load_data()
    preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer()

    results = {}
    for scaler_name in ['standard', 'robust']:
        X_scaled = preprocessor.scale_data(X, scaler_name)
        results[scaler_name] = model_trainer.train_evaluate(X_scaled, y)

    return results, X, y


def run_phase2(X, y, best_scaler):
    """Phase 2: Feature engineering"""
    preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer()

    X_scaled = preprocessor.scale_data(X, best_scaler)
    results = {}

    # Test PCA
    for pca_name in ['pca_50', 'pca_100']:
        X_reduced = preprocessor.reduce_dimensions(X_scaled, pca_name)
        results[pca_name] = model_trainer.train_evaluate(X_reduced, y)

    # Test Feature Selection
    feature_selectors = preprocessor.create_feature_selectors(
        model_trainer.classifiers['rf']
    )
    for selector_name, selector in feature_selectors.items():
        X_selected = selector.fit_transform(X_scaled, y)
        results[selector_name] = model_trainer.train_evaluate(X_selected, y)

    return results, X_scaled


def run_phase3(X, y, best_preprocessor):
    """Phase 3: Classifier optimization"""
    model_trainer = ModelTrainer()
    results = {}

    for clf_name in ['rf', 'svm', 'knn', 'ada']:
        results[clf_name] = model_trainer.train_evaluate(X, y, clf_name)

    return results


def run_phase4(X, y):
    """Phase 4: Ensemble methods"""
    model_trainer = ModelTrainer()
    results = {}

    for method in ['bagging', 'adaboost']:
        results[method] = model_trainer.train_ensemble(X, y, method)

    return results


def main():
    try:
        # Phase 1
        print("Phase 1: Testing scalers")
        phase1_results, X, y = run_phase1()
        best_scaler = find_best_method(phase1_results)

        # Phase 2
        print("\nPhase 2: Feature engineering")
        phase2_results, X_scaled = run_phase2(X, y, best_scaler)
        best_preprocessor = find_best_method(phase2_results)

        # Phase 3
        print("\nPhase 3: Classifier optimization")
        phase3_results = run_phase3(X_scaled, y, best_preprocessor)

        # Phase 4
        print("\nPhase 4: Ensemble methods")
        phase4_results = run_phase4(X_scaled, y)

        # Collect all results
        results = {
            'phase1': phase1_results,
            'phase2': phase2_results,
            'phase3': phase3_results,
            'phase4': phase4_results
        }

        # Print results
        print_results(results)
        return results

    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()