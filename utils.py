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

            if file.endswith(('.wav')):
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
    audio, sr = librosa.load(file_path, sr=sr, duration=duration)
    
    # consistent length
    target_length = int(sr * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # mel spectogram
    # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    ) 

    # decibel conversion
    spectrogram_dB = librosa.power_todb(spectrogram, ref=np.max)

    # normalize
    spectrogram_dB = (spectrogram_dB - spectrogram_dB.min()) / (spectrogram_dB.max() - spectrogram_dB.min())

    return spectrogram_dB

# creating dataset
def data_set(data_dir, batch_size, target_shape, shuffle=True):
    """
    Creates dataset from the audio files
    """

    audio_files = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for audio_file in os.listdir(class_dir):
            if audio_file.endswith(('.wav')):
                audio_files.append(os.path.join(class_dir, audio_file))
                labels.append(class_idx)

    def generator():
        for audio_file, label in zip(audio_files, labels):
            spe = process_audio(audio_file)
            if spec is not None:
                spec = tf.image.resize(spec[..., np.newaxis], target_shape)
                yield spec, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(target_shape[0], target_shape[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset