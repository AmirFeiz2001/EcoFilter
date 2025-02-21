from keras.applications.vgg16 import VGG16
from keras.models import Model
import xgboost as xgb

def build_feature_extractor(input_shape=(150, 150, 3)):
    """
    Build VGG16 model for feature extraction.

    Args:
        input_shape (tuple): Input shape of images.

    Returns:
        Model: VGG16 model without top layers.
    """
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg_model.layers:
        layer.trainable = False
    return vgg_model

def extract_features(model, images):
    """
    Extract features using the VGG16 model.

    Args:
        model (Model): VGG16 feature extractor.
        images (np.ndarray): Images to process.

    Returns:
        np.ndarray: Flattened feature vectors.
    """
    return model.predict(images).reshape(images.shape[0], -1)

def build_classifier():
    """
    Build an XGBoost classifier.

    Returns:
        xgb.XGBClassifier: XGBoost model.
    """
    return xgb.XGBClassifier()
