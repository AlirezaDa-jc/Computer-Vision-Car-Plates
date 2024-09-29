import numpy as np
from sklearn.metrics import mean_squared_error
import cv2
from keras.models import load_model
from loader import load, resize

def corners(image: np.ndarray, model) -> np.ndarray:
    image, _ = resize(image, np.zeros((4, 2)))
    image = np.expand_dims(image, axis=0)
    points = model.predict(image)[0]
    return points.reshape(4, 2)
