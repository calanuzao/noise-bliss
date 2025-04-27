"""
utils.py

Helper functions and classes for:
- Loading, processing, and augmenting audio data
- Dataset handling with TensorFlow and Soundata
- Feature extraction (Mel spectrograms)
- Data visualization (temporal and location distributions)
"""

# ==================== Library Imports ==================== #
# Install additional libraries (if needed)
# %pip install tensorflow-io  # Uncomment if tensorflow-io not installed

# Core Libraries
import tensorflow as tf            # TensorFlow for deep learning
import tensorflow_io as tfio        # TensorFlow I/O extensions
from tensorflow import keras        # Keras high-level API
from keras import layers            # Keras layers
from tensorflow.keras.models import model_from_json  # Load Keras model from JSON

# Custom Modules
import utils as u                   # Custom utilities module
import models as m                  # Custom models module

# Visualization and Display
import matplotlib.pyplot as plt     # Plotting graphs and spectrograms
import IPython.display as ipd       # For displaying audio inside Jupyter notebooks
import datetime
import tqdm
from tqdm import tqdm


# File Handling and Data Processing
import os                           # Operating system utilities
import pandas as pd                 # Data handling (DataFrames)
import numpy as np                  # Numerical operations
import random                       # Random number generation
from pathlib import Path            # Path handling

# Audio Processing
import librosa                      # Audio processing (loading, spectrograms)
import librosa.display              # Spectrogram display utilities
import soundata                     # Standardized dataset loading (UrbanSound, SONYC)
from soundata.core import Dataset

# Image Processing
import cv2                           # Image processing (used for resizing spectrograms)

# Specialized Architectures
from tcn import TCN                  # Temporal Convolutional Networks (sequential modeling)

# Machine Learning Metrics
from sklearn.metrics import classification_report  # Classification metrics (precision, recall, f1)

# TensorFlow Dataset Prefetching
AUTOTUNE = tf.data.AUTOTUNE           # Optimize tf.data pipelines


# ==================== Constants ==================== #

eps = np.finfo(float).eps
AUTOTUNE = tf.data.AUTOTUNE

label_taxonomy = {
    'engine': 0,
    'machinery-impact': 1,
    'non-machinery-impact': 2,
    'powered-saw': 3,
    'alert-signal': 4,
    'music': 5,
    'human-voice': 6,
    'dog': 7
}

# ==================== Dataset Handling ==================== #

def prepare_dataset(file_paths, labels, batch_size, training=True):
    def load_and_preprocess(file_path, label):
        audio, _ = tf.numpy_function(librosa.load, [file_path, 22050], [tf.float32, tf.int32])
        audio.set_shape([None])
        mel_spec = tf.numpy_function(lambda x: librosa.feature.melspectrogram(x.numpy(), sr=22050, n_mels=64), [audio], tf.float32)
        mel_spec.set_shape([64, None])
        return mel_spec, label

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size=5000)

    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    return dataset

class SonycUSTDataset(Dataset):
    """Custom Soundata dataset class for SONYC-UST."""

    def __init__(self, data_home):
        self.data_home = data_home
        self.annotations_path = os.path.join(data_home, 'annotations.csv')
        self.audio_dir = os.path.join(data_home, 'audio')

    def load_audio(self, audio_file):
        audio_path = os.path.join(self.audio_dir, audio_file)
        return soundata.load_audio(audio_path)

    def load_annotations(self):
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found at {self.annotations_path}")
        return pd.read_csv(self.annotations_path)
    


def load_data(data_home):
    """
    Load SONYC-UST dataset: audio files and multi-hot encoded labels.

    Parameters
    ----------
    data_home : str
        The root directory containing the SONYC-UST dataset.

    Returns
    -------
    audio_file_paths : list of str
        List of paths to audio files.
    labels : list of list of int
        List of multi-hot encoded labels for each audio file.
    """
    try:
        audio_dir = os.path.join(data_home, 'audio')
        annotations_path = os.path.join(data_home, 'annotations.csv')

        if not os.path.exists(audio_dir):
            raise ValueError(f"Audio directory not found: {audio_dir}")
        if not os.path.exists(annotations_path):
            raise ValueError(f"Annotations file not found: {annotations_path}")

        annotations_df = pd.read_csv(annotations_path)

        column_mapping = {
            'engine': '1_engine_presence',
            'machinery-impact': '2_machinery-impact_presence',
            'non-machinery-impact': '3_non-machinery-impact_presence',
            'powered-saw': '4_powered-saw_presence',
            'alert-signal': '5_alert-signal_presence',
            'music': '6_music_presence',
            'human-voice': '7_human-voice_presence',
            'dog': '8_dog_presence'
        }

        audio_file_paths = []
        labels = []

        unique_files = annotations_df['audio_filename'].unique()

        # Add tqdm progress bar here
        for audio_file in tqdm(unique_files, desc="Loading audio files"):
            file_path = os.path.join(audio_dir, audio_file)
            if os.path.exists(file_path):
                file_annotations = annotations_df[annotations_df['audio_filename'] == audio_file]

                label = [0] * len(label_taxonomy)
                for sound_category, column_name in column_mapping.items():
                    values = file_annotations[column_name].values
                    if 1 in values:
                        label[label_taxonomy[sound_category]] = 1

                audio_file_paths.append(file_path)
                labels.append(label)

        if not audio_file_paths:
            raise ValueError(f"No valid audio files found in {audio_dir}")

        return audio_file_paths, labels

    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# ==================== Audio Processing ==================== #
def augment_spectrogram(spec):
    """
    Augments a mel spectrogram with random frequency and time masking.

    Parameters:
    ----------
    spec : np.ndarray
        Mel spectrogram to be augmented.
    
    Returns:
    -------
    np.ndarray
        Augmented mel spectrogram.
    """
    # Frequency masking
    freq_mask = tf.random.uniform([], 0, 20, dtype=tf.int32)
    spec = tf.roll(spec, freq_mask, axis=0)

    # Time masking
    time_mask = tf.random.uniform([], 0, 20, dtype=tf.int32)
    spec = tf.roll(spec, time_mask, axis=1)

    return spec

def extract_audio_only(dataset):
    """Patch a fusion dataset to yield only audio_input."""
    def generator():
        for x, y in dataset:
            yield x['audio_input'], y

    output_signature = (
        tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(8,), dtype=tf.float32)
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

def augment_audio(waveform, sample_rate):
    waveform_aug = waveform.copy()

    noise_factor = 0.005
    noise = np.random.randn(*waveform_aug.shape)
    waveform_aug += noise_factor * noise

    shift_max = int(0.1 * sample_rate)
    shift = np.random.randint(-shift_max, shift_max)
    waveform_aug = np.roll(waveform_aug, shift)

    if np.random.rand() < 0.5:
        stretch_rate = np.random.uniform(0.8, 1.2)
        waveform_aug = librosa.effects.time_stretch(waveform_aug.squeeze(), rate=stretch_rate)
        if len(waveform_aug.shape) == 1:
            waveform_aug = waveform_aug[np.newaxis, :]

    return waveform_aug

def process_audio(file_path, sr=22050, duration=None, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        if audio is None or len(audio) == 0:
            print(f"Warning: Failed to load audio from {file_path}")
            return None

        if augment:
            audio = augment_audio(audio[np.newaxis, :], sr).squeeze()

        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_dB = (spectrogram_dB - spectrogram_dB.min()) / (spectrogram_dB.max() - spectrogram_dB.min() + 1e-6)

        return spectrogram_dB
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# ==================== Feature Extraction ==================== #

def mel_spectrogram(audio, sample_rate=22050, n_mels=128, hop_length=512):
    audio = librosa.to_mono(audio)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length=hop_length, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(mel_spectrogram)

    librosa.display.specshow(log_spectrogram, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

# ==================== Data Visualization ==================== #

def plot_distribution(data, xlabel, ylabel, title, kind='bar', xticks=None):
    plt.figure(figsize=(10, 4))
    if kind == 'bar':
        plt.bar(data.index, data.values, color='steelblue')
    elif kind == 'line':
        plt.plot(data.index, data.values, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xticks is not None:
        plt.xticks(xticks)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# ==================== Temporal and Location Distributions ==================== #

def compute_and_plot_distributions(df):
    """
    Plots hourly, daily, weekly, monthly, and yearly distributions
    of the 'datetime' field across the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'datetime' column.
    """

    # Compute distributions
    hourly = df['datetime'].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
    daily = df['datetime'].dt.date.value_counts().sort_index()
    weekly = df['datetime'].dt.isocalendar().week.value_counts().sort_index()
    monthly = df['datetime'].dt.month.value_counts().reindex(range(1, 13), fill_value=0).sort_index()
    yearly = df['datetime'].dt.year.value_counts().sort_index()

    # Plot distributions
    plot_distribution(hourly, "Hour of Day", "Recordings", "Hourly Distribution", xticks=range(24))
    plot_distribution(daily, "Date", "Recordings", "Daily Distribution", kind='line')
    plot_distribution(weekly, "ISO Week", "Recordings", "Weekly Distribution")
    plot_distribution(monthly, "Month", "Recordings", "Monthly Distribution", xticks=range(1, 13))
    plot_distribution(yearly, "Year", "Recordings", "Yearly Distribution")

def plot_distributions_by_split(df):
    split_colors = {
        'train': '#E69F00',
        'validate': '#56B4E9',
        'test': '#009E73'
    }

    time_granularities = {
        'Hourly': df['datetime'].dt.hour,
        'Daily': df['datetime'].dt.date,
        'Weekly': df['datetime'].dt.isocalendar().week,
        'Monthly': df['datetime'].dt.month,
        'Yearly': df['datetime'].dt.year
    }

    for name, series in time_granularities.items():
        plt.figure(figsize=(12, 4))
        for split in df['split'].unique():
            subset = df[df['split'] == split]
            counts = series[subset.index].value_counts().sort_index()

            plt.plot(counts.index, counts.values, label=split.capitalize(), color=split_colors.get(split, 'black'), linewidth=1)

        plt.title(f"{name} Distribution by Split")
        plt.xlabel(name)
        plt.ylabel("Number of Recordings")
        plt.legend(title="Split")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# ==================== Temporal Metadata Construction ==================== #

def construct_datetime(df):
    dt_series = df.apply(lambda row: datetime.datetime.strptime(
        f"{int(row['year'])} {int(row['week']) + 1} {int(row['day']) + 1}",
        "%G %V %u"
    ) + pd.Timedelta(hours=int(row['hour'])), axis=1)

    df['datetime'] = pd.to_datetime(dt_series)
    return df

def construct_location_id(df):
    df['location_id'] = (
        df['borough'].astype(str) + "_" +
        df['block'].astype(str) + "_" +
        df['sensor_id'].astype(str)
    )
    return df

# ==================== Updated Dataset Creation ==================== #

def data_set(file_paths, batch_size, target_shape, annotations_df, shuffle=True, augment=False):
    def process_audio_file(file_path):
        try:
            audio, sr = librosa.load(file_path, sr=48000)
            if augment:
                audio = augment_audio(audio[np.newaxis, :], sr).squeeze()

            spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=target_shape[0],
                hop_length=512
            )
            spec_db = librosa.power_to_db(spec, ref=np.max)
            spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)

            if spec_norm.shape != target_shape:
                spec_norm = cv2.resize(spec_norm, (target_shape[1], target_shape[0]))

            spec_norm = np.expand_dims(spec_norm, axis=-1)
            return spec_norm.astype(np.float32)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def get_label(audio_path):
        try:
            filename = os.path.basename(audio_path)
            file_annotations = annotations_df[annotations_df['audio_filename'] == filename]
            
            if file_annotations.empty:
                print(f"No annotations found for {filename}")
                return None
                
            label = [0] * len(label_taxonomy)
            for category, idx in label_taxonomy.items():
                column = f"{idx+1}_{category}_presence"
                if column in file_annotations.columns and (file_annotations[column] == 1).any():
                    label[idx] = 1
            return np.array(label, dtype=np.float32)
        except Exception as e:
            print(f"Error getting label for {audio_path}: {str(e)}")
            return None

    def generator():
        valid_files = 0
        for file_path in file_paths:
            try:
                spec = process_audio_file(file_path)
                if spec is not None:
                    label = get_label(file_path)
                    if label is not None:
                        valid_files += 1
                        if shuffle:  # shuffle indicates training data
                            spec = augment_spectrogram(spec)
                    yield spec, label
            except Exception as e:
                continue

        if valid_files == 0:
            raise ValueError("No valid audio files processed.")

    output_signature = (
        tf.TensorSpec(shape=(target_shape[0], target_shape[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(len(label_taxonomy),), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset