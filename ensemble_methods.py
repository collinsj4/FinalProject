from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

def bagging_classifier(X_train, y_train, X_test, base_model):
    """Trains and evaluates a Bagging classifier."""
    bagging = BaggingClassifier(base_estimator=base_model, random_state=42)
    bagging.fit(X_train, y_train)
    return bagging.predict(X_test)

def adaboost_classifier(X_train, y_train, X_test, base_model):
    """Trains and evaluates an AdaBoost classifier."""
    adaboost = AdaBoostClassifier(base_estimator=base_model, random_state=42)
    adaboost.fit(X_train, y_train)
    return adaboost.predict(X_test)
