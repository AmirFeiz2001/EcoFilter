import argparse
import numpy as np
from src.preprocess import load_and_split_data, encode_labels, normalize_images, apply_registration
from src.model import build_feature_extractor, extract_features, build_classifier
from src.train import train_model, evaluate_model
from src.visualize import plot_confusion_matrix, compare_images

def main():
    parser = argparse.ArgumentParser(description="Image Classification with VGG16 and XGBoost")
    parser.add_argument('--dataset', type=str, required=True, help='Path to numpy array of images')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    args = parser.parse_args()

    # Load dataset (assuming dataset and labels are numpy arrays or lists)
    dataset = np.load(args.dataset)  # Adjust based on your data format
    labels = np.load(args.labels)    # Adjust based on your data format

    # Preprocess
    x_train, x_test, train_labels, test_labels = load_and_split_data(dataset, labels)
    train_labels_encoded, test_labels_encoded, le = encode_labels(train_labels, test_labels)
    x_train, x_test = normalize_images(x_train, x_test)
    x_train = apply_registration(x_train, x_train[0])
    x_test = apply_registration(x_test, x_test[0])

    # Feature extraction
    vgg_model = build_feature_extractor()
    X_train_features = extract_features(vgg_model, x_train)
    X_test_features = extract_features(vgg_model, x_test)

    # Train and evaluate
    classifier = build_classifier()
    trained_model = train_model(classifier, X_train_features, train_labels_encoded, args.output_dir)
    accuracy, predictions = evaluate_model(trained_model, X_test_features, test_labels, le)

    # Visualize
    plot_confusion_matrix(test_labels, predictions, args.output_dir + "/plots")
    compare_images(x_test[0], register_images(x_test[0], x_test[1]), "Image Registration Comparison", args.output_dir + "/plots")

if __name__ == "__main__":
    main()
