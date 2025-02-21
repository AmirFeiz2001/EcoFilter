import os
import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import shift
from sklearn.preprocessing import LabelEncoder

def load_and_split_data(dataset, labels, train_size=800, img_size=150):
    train_images = dataset[:train_size]
    test_images = dataset[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]
    
    return np.array(train_images), np.array(test_images), train_labels, test_labels

def encode_labels(train_labels, test_labels):
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    test_labels_encoded = le.transform(test_labels)
    
    return train_labels_encoded, test_labels_encoded, le

def normalize_images(train_images, test_images):
    return train_images / 255.0, test_images / 255.0

def register_images(ref_image, target_image):
    try:
        shift_values, _, _ = register_translation(ref_image, target_image, 100)
        corrected_image = shift(target_image, shift=(shift_values[0], shift_values[1], 0), mode='constant')
        return corrected_image
    except Exception as e:
        print(f"Error in image registration: {e}")
        
        return target_image

def apply_registration(images, ref_image):
    
    return np.array([register_images(ref_image, img) for img in images])
