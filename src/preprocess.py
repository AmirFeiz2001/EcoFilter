import os
import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import shift
from sklearn.preprocessing import LabelEncoder

def load_and_split_data(dataset, labels, train_size=800, img_size=150):
    """
    Load and split dataset into train and test sets.

    Args:
        dataset (list): List of images.
        labels (list): List of corresponding labels.
        train_size (int): Number of samples for training.
        img_size (int): Target image size.

    Returns:
        tuple: Train and test images and labels.
    """
    train_images = dataset[:train_size]
    test_images = dataset[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]
    return np.array(train_images), np.array(test_images), train_labels, test_labels

def encode_labels(train_labels, test_labels):
    """
    Encode categorical labels to numerical values.

    Args:
        train_labels (list): Training labels.
        test_labels (list): Test labels.

    Returns:
        tuple: Encoded labels and LabelEncoder object.
    """
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    test_labels_encoded = le.transform(test_labels)
    return train_labels_encoded, test_labels_encoded, le

def normalize_images(train_images, test_images):
    """
    Normalize image pixel values to [0, 1].

    Args:
        train_images (np.ndarray): Training images.
        test_images (np.ndarray): Test images.

    Returns:
        tuple: Normalized training and test images.
    """
    return train_images / 255.0, test_images / 255.0

def register_images(ref_image, target_image):
    """
    Register a target image to a reference image.

    Args:
        ref_image (np.ndarray): Reference image.
        target_image (np.ndarray): Image to align.

    Returns:
        np.ndarray: Corrected (registered) image.
    """
    try:
        shift_values, _, _ = register_translation(ref_image, target_image, 100)
        corrected_image = shift(target_image, shift=(shift_values[0], shift_values[1], 0), mode='constant')
        return corrected_image
    except Exception as e:
        print(f"Error in image registration: {e}")
        return target_image  # Return original if registration fails

def apply_registration(images, ref_image):
    """
    Apply image registration to a set of images.

    Args:
        images (np.ndarray): Array of images to register.
        ref_image (np.ndarray): Reference image for alignment.

    Returns:
        np.ndarray: Registered images.
    """
    return np.array([register_images(ref_image, img) for img in images])
