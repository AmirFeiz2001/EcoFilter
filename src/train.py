import os
from sklearn.metrics import accuracy_score

def train_model(classifier, X_train, y_train, output_dir="output"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classifier.fit(X_train, y_train)
    model_path = os.path.join(output_dir, "XGBoost_EcoFilter.json")
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    return classifier

def evaluate_model(classifier, X_test, y_test, le):

    predictions = classifier.predict(X_test)
    predictions = le.inverse_transform(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy, predictions
