from keras.applications.vgg16 import VGG16
from keras.models import Model
import xgboost as xgb

def build_feature_extractor(input_shape=(150, 150, 3)):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg_model.layers:
        layer.trainable = False
    return vgg_model

def extract_features(model, images):

    return model.predict(images).reshape(images.shape[0], -1)

def build_classifier():
    return xgb.XGBClassifier()
