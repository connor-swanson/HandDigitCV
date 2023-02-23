import cv2
import numpy as np
from typing import Tuple
from sklearn import preprocessing
import tensorflow as tf


def make_prediction(model, le, raw_image):

    pred = model.predict(np.array(raw_image).reshape(1, 128, 128, 3))

    predicted_classes = np.argmax(np.round(pred), axis=1)

    return le.inverse_transform(predicted_classes)


def load_models():
    # encoder
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load('./trained_models/le.npy')

    # predicted
    model = tf.keras.models.load_model('./trained_models/m2')

    return model, encoder


def draw_label(
    img: np.array,
    display_text: str,
    display_position: Tuple[int, int] = (50, 50),
    display_color: Tuple[int, int, int] = (255, 0, 0),
    font_scale: float = 1.0,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    thickness=cv2.FILLED,
    margin: int = 10
):

    txt_size = cv2.getTextSize(display_text, font_face, font_scale, thickness)

    end_x = display_position[0] + txt_size[0][0] + margin
    end_y = display_position[1] - txt_size[0][1] - margin

    # draw box
    cv2.rectangle(img, display_position, (end_x, end_y), display_color, thickness)

    # Overlay Prediction
    cv2.putText(img, display_text, display_position, font_face, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
