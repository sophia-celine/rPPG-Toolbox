import cv2
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import re

import torch

import numpy as np
# import ipywidgets as widgets
from IPython.display import display, clear_output
from natsort import natsorted
import os
import scipy
from scipy.sparse import spdiags
from scipy.signal import butter
import math
from scipy import linalg
from scipy import signal
from scipy import sparse


data_out_path = "/home/soph/rppg/rPPG-Toolbox/runs/exp/test_SizeW128_SizeH128_ClipLength160_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxFalse_Large_size1.5_Dyamic_DetTrue_det_len25_Median_face_boxFalse/saved_test_outputs/PURE_PhysFormer_DiffNormalized_test_outputs.pickle"
chunk_size = 360 # size of chunk to visualize: -1 will plot the entire signal
chunk_num = 0
model = "PhysFormerPURE"
path = "/home/soph/rppg/rPPG-Toolbox/BVPresults"

# HELPER FUNCTIONS

def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data

def _process_signal(signal, fs=30, diff_flag=True, use_bandpass=False):
    # Filter
    if use_bandpass:
        [b, a] = butter(3, [0.2 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        signal = scipy.signal.filtfilt(b, a, np.double(signal))
    return signal

# def _detrend(input_signal, lambda_value):
#     """Detrend PPG signal."""
#     signal_length = input_signal.shape[0]
#     # observation matrix
#     H = np.identity(signal_length)
#     ones = np.ones(signal_length)
#     minus_twos = -2 * np.ones(signal_length)
#     diags_data = np.array([ones, minus_twos, ones])
#     diags_index = np.array([0, 1, 2])
#     D = spdiags(diags_data, diags_index,
#                 (signal_length - 2), signal_length).toarray()
#     detrended_signal = np.dot(
#         (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
#     return detrended_signal

# Read in data and list subjects
with open(data_out_path, 'rb') as f:
    data = pickle.load(f)
    
# List of all video trials
trial_list = list(data['predictions'].keys())
print(trial_list)
for trial_idx in range(len(trial_list)):
    prediction = np.array(_reform_data_from_dict(data['predictions'][trial_list[trial_idx]]))
    label = np.array(_reform_data_from_dict(data['labels'][trial_list[trial_idx]]))

    fs = data['fs'] # Video Frame Rate
    label_type = data['label_type'] # PPG Signal Transformation: `DiffNormalized` or `Standardized`
    diff_flag = (label_type == 'DiffNormalized')

    if chunk_size == -1:
        chunk_size = len(prediction)
        chunk_num = 0

    # Process label and prediction signals
    prediction = _process_signal(prediction, fs, diff_flag=diff_flag)

    np.savetxt(f"../../BVPresults/BVP_{model}_{trial_list[trial_idx]}.txt", prediction, fmt='%.7e')
    print(f"{path}/BVP_{model}_{trial_list[trial_idx]}.txt")
    np.savetxt(f"{path}/BVP_{model}_{trial_list[trial_idx]}.txt", prediction, fmt='%.7e')

    # plt.plot(prediction)
    # plt.show()