from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline


class ModelTrainer:
    def __init__(self):
        self.classifiers = {
            'rf': RandomForestClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(),
            'ada': AdaBoostClassifier(random_state=42)
        }

        self.cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    def train_evaluate(self, X, y, classifier_name='rf'):
        """Train and evaluate a classifier"""
        try:
            if isinstance(classifier_name, str):
                clf = self.classifiers[classifier_name]
            else:
                clf = classifier_name

            metrics = {
                'accuracy': cross_val_score(clf, X, y, cv=self.cv, scoring='accuracy').mean(),
                'f1': cross_val_score(clf, X, y, cv=self.cv, scoring='f1_weighted').mean(),
                'roc_auc': cross_val_score(clf, X, y, cv=self.cv, scoring='roc_auc_ovr').mean()
            }

            return metrics
        except Exception as e:
            print(f"Error in train_evaluate: {e}")
            return {'accuracy': 0, 'f1': 0, 'roc_auc': 0}

    def train_ensemble(self, X, y, method='bagging'):
        """Train and evaluate an ensemble classifier"""
        try:
            # Use DecisionTreeClassifier as base estimator
            base_estimator = DecisionTreeClassifier(random_state=42)

            if method == 'bagging':
                ensemble = BaggingClassifier(
                    estimator=base_estimator,
                    n_estimators=10,
                    random_state=42
                )
            else:
                ensemble = AdaBoostClassifier(
                    estimator=base_estimator,
                    n_estimators=10,
                    random_state=42
                )

            return self.train_evaluate(X, y, ensemble)

        except Exception as e:
            print(f"Error in ensemble method {method}: {e}")
            return {'accuracy': 0, 'f1': 0, 'roc_auc': 0}