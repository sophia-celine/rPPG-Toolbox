import math

import cv2
import numpy as np
from scipy import io as scio
from scipy import linalg
from scipy import signal
from scipy import sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def process_video(frames):
    """Return per-frame RGB means as (1, 3, T), optionally using a mask channel."""
    RGB = []
    for frame in frames:
        channels = frame[..., :3]
        if frame.shape[-1] > 3:
            mask = frame[..., 3] > 0
            if not np.any(mask):
                RGB.append(np.zeros(3, dtype=np.float32))
                continue
            RGB.append(np.mean(channels[mask], axis=0))
        else:
            RGB.append(np.mean(channels, axis=(0, 1)))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)
