from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import random
import librosa
import librosa.display
import cv2
import os
import datetime
import pandas as pd
from pathlib import Path

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
        and the second array contains their corresponding labels.
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

# temporal data
def construct_datetime(df):
    """
    Constructs a 'datetime' column from 'year'. 'week', 'day', 'hour'
    Retunrs the modified DataFrame.
    """
    dt_series = df.apply(lambda row: datetime.datetime.strptime(
        f"{int(row['year'])} {int(row['week']) + 1} {int(row['day']) + 1}",
        "%G %V %u"                                      
    ) + pd.Timedelta(hours=int(row['hour'])), axis=1) 

    df['datetime'] = pd.to_datetime(dt_series)
    return df

# location id
def construct_location_id(df):
    """
    A unique 'location_id' column from 'sensor_id', 'borough', and 'block'.
    Returns the modified DataFrame.
    """
    df['location_id'] = (
        df['borough'].astype(str) + "_" +
        df['block'].astype(str) + "_" +
        df['sensor_id'.astype(str)]
    )
    return df

# creating dataset
def data_set(file_list, batch_size, target_shape, shuffle=True):
    """
    Creates dataset from the list of audio files
    """
    def generator():
        for audio_file in file_list:
            spec = process_audio(audio_file)
            if spec is not None:
                spec = tf.image.resize(spec[..., np.newaxis], target_shape)
                yield spec

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(
                shape=(target_shape[0], target_shape[1], 1),
                dtype=tf.float32       
            )
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

### ============= Plotting ============= ###

def plot_distribution(data, xlabel, ylabel, title, kind='bar', xticks=None):
    plt.figure(figsize=(10,4))
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

def compute_and_plot_distributions(df):
    """
    Plots hourly, daily, weekly, monthly, and yearly distributions.
    """
    hourly = df['datetime'].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
    daily = df['datetime'].dt.date.value_counts().sort_index()
    weekly = df['datetime'].dt.isocalendar().week.value_counts().sort_index()
    monthly = df['datetime'].dt.month.value_counts().reindex(range(1, 13), fill_value=0).sort_index()
    yearly = df['datetime'].dt.year.value_counts().sort_index()

    plot_distribution(hourly, "Hour of Day", "Recordings", "Hourly Distribution", xticks=range(24))
    plot_distribution(daily, "Date", "Recordings", "Daily Distribution", kind='line')
    plot_distribution(weekly, "ISO Week", "Recordings", "Weekly Distribution")
    plot_distribution(monthly, "Month", "Recordings", "Monthly Distribution", xticks=range(1, 13))
    plot_distribution(yearly, "Year", "Recordings", "Yearly Distribution")

def compute_and_plot_location_distributions(df):
    """
    Plots sensor-based and borough-based distributions.
    Assumes 'location_id' and 'borough' columns exists
    """
    # count by sensor
    sensor_counts = df['sensor_id'].value_counts().sort_index()

    # count by borough
    borough_counts = df['borough'].value_counts().sort_index()

    # count by full location_id
    location_counts = df['location_id'].value_counts().sort_index()

    plot_distribution(sensor_counts, "Sensor ID", "Recordings", "Recording per Sensor")
    plot_distribution(borough_counts, "Borough Code", "Recordings", "Recording per Borough")
    plot_distribution(location_counts, "Locaiton ID", "Recordings", "Recording per Unique Location", kind='bar')

def plot_distributions_by_split(df):
    """
    Plots hourly, daily, weekly, monthly, and yearly distribution by split.
    """
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

            plt.plot(
                counts.index,
                counts.values,
                label=split.capitalize(),
                color=split_colors.get(split, 'black'),
                linewidth=1
            )

        plt.title(f"{name} Distribution by Split")
        plt.xlabel(name)
        plt.ylabel("Number of Recordings")
        plt.legend(title="Split")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

def mel_spectrogram(audio, sample_rate=2250, n_mels=128, hop_length=512):
    """
    Computes the normalized Mel spectrogram of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal as a 1D numpy array.
    sample_rate : int, optional
        Sampling rate of the audio signal, by default 22050.
    n_mels : int, optional
        Number of Mel bands to generate, by default 128.
    hop_length : int, optional
        Number of samples between successive frames, by default 512.

    Returns
    -------
    np.ndarray
        Mel spectrogram as a 2D numpy array.
    """
    audio = librosa.to_mono(audio)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,                
        sr=sample_rate,         
        hop_length=hop_length,  
        n_mels=n_mels           
    )

    log_spectrogram = librosa.power_to_db(mel_spectrogram)
    librosa.display.specshow(
        log_spectrogram,
        x_axis='time',
        y_axis='mel', 
        sr=sample_rate, 
        fmax=8000
        )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()