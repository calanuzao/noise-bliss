from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import random
import librosa
import cv2
import os

### ============= Data Generation Functions ============= ###

def load_data(data_path):
    """
    Load audio files and their corresponding characteristics.

    Parameters:
    data_path: str
            Path to the directory containing the SONYC-UST Data

    Returns:
    Tuple[np.ndarray, np.ndarray]
        A tuple of the two numpy arrays.
        The first array contains the file paths of the audio files, 
        and the second array contains their corresponding lables.
    """

    if isinstance(data_path, bytes):
        data_path = data_path.decode("utf-8")

    audio_files = []
    labels = []

    for root, dirs, files in os.walk(data_path):
        for file in files:

            if file.endswith(('.wav', '.mp3', '.ogg')):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)

                label = os.path.basename(root)
                if isinstance(label, bytes):
                    labels = label.decode('utf8')
                labels.append(label)

    return np.array(audio_files), np.array(labels)
            
# Loading and processing audio files
def process_audio(file_path, sr=22050, duration=None):
    """
    Load and processes an audio file.

    Parameter:
    file_path: str
        Path to the audio file
    sr: int
        Sampling rate (libros sr is set to 22050)
    duration: float or None
        Duration in seconds to load, or None for full file

    Returns:
    np.ndarray
        Processed audio data
    """
    audio, sr = librosa.load(file_path, ssr=sr, duration=duration)
    # add more processing steps

    return audio
