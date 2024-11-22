from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def knn_classifier(X_train, y_train, X_test, k=3):
    """Trains and evaluates a KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

def logistic_regression_classifier(X_train, y_train, X_test):
    """Trains and evaluates a Logistic Regression classifier."""
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def svm_classifier(X_train, y_train, X_test, kernel="linear"):
    """Trains and evaluates an SVM classifier."""
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def random_forest_classifier(X_train, y_train, X_test):
    """Trains and evaluates a Random Forest classifier."""
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
