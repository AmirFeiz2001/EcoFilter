import os
from sklearn.metrics import accuracy_score

def train_model(classifier, X_train, y_train, output_dir="output"):
    """
    Train the XGBoost classifier.

    Args:
        classifier (xgb.XGBClassifier): XGBoost model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        output_dir (str): Directory to save the model.

    Returns:
        xgb.XGBClassifier: Trained model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classifier.fit(X_train, y_train)
    model_path = os.path.join(output_dir, "XGBoost_EcoFilter.json")
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    return classifier

def evaluate_model(classifier, X_test, y_test, le):
    """
    Evaluate the model on test data.

    Args:
        classifier (xgb.XGBClassifier): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        le (LabelEncoder): Label encoder for decoding predictions.

    Returns:
        float: Accuracy score.
    """
    predictions = classifier.predict(X_test)
    predictions = le.inverse_transform(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy, predictions
