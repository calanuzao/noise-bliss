from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tcn import TCN
import tensorflow as tf
import tensorflow_io as tfio
import keras
from tensorflow import keras 
from keras import layers
from tensorflow.keras.models import model_from_json
import random
import librosa
import librosa.display
import cv2
import os
import datetime
from pathlib import Path
import soundata
from soundata.core import Dataset
import pandas as pd
import os
import numpy as np

eps = np.finfo(float).eps

# https://zenodo.org/records/3966543
label_taxonomy = {
    'engine': 0,
    'machinery-impact': 1,
    'non-machinery-impact': 2,
    'powered-saw': 3,
    'alert-signal': 4,
    'music':5,
    'human-voice': 6,
    'dog': 7
}

### ============= Data Generation Functions ============= ###
class SonycUSTDataset(Dataset):
    """Custom soundata dataset for SONYC-UST"""
    
    def __init__(self, data_home):
        self.data_home = data_home
        self.annotations_path = os.path.join(data_home, 'annotations.csv')
        self.audio_dir = os.path.join(data_home, 'audio')
        
    def load_audio(self, audio_file):
        """Load audio file"""
        audio_path = os.path.join(self.audio_dir, audio_file)
        return soundata.load_audio(audio_path)
        
    def load_annotations(self):
        """Load annotations from CSV file"""
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found at {self.annotations_path}")
        return pd.read_csv(self.annotations_path)

def load_data(data_home):
    """
    Load SONYC-UST dataset.

    Parameters
    ----------
    data_home : str
        The root directory containing the SONYC-UST dataset

    Returns
    -------
    audio_file_paths : list of str
        List of paths to audio files
    labels : list of list
        List of multi-hot encoded labels for each audio file
    """
    try:
        # Get paths
        audio_dir = os.path.join(data_home, 'audio')
        annotations_path = os.path.join(data_home, 'annotations.csv')
        
        if not os.path.exists(audio_dir):
            raise ValueError(f"Audio directory not found: {audio_dir}")
        if not os.path.exists(annotations_path):
            raise ValueError(f"Annotations file not found: {annotations_path}")
            
        # Load annotations
        annotations_df = pd.read_csv(annotations_path)
        
        # Map taxonomy to column names
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
        
        # Process unique audio files (avoid duplicates from multiple annotators)
        unique_files = annotations_df['audio_filename'].unique()
        
        for audio_file in unique_files:
            file_path = os.path.join(audio_dir, audio_file)
            if os.path.exists(file_path):
                # Get all annotations for this file
                file_annotations = annotations_df[annotations_df['audio_filename'] == audio_file]
                
                # Create multi-hot encoded label
                label = [0] * len(label_taxonomy)
                
                # For each category in our taxonomy
                for sound_category, column_name in column_mapping.items():
                    # Get all values for this category (from different annotators)
                    values = file_annotations[column_name].values
                    # If any annotator marked it as present (1), count it as present
                    if 1 in values:
                        label[label_taxonomy[sound_category]] = 1
                
                audio_file_paths.append(file_path)
                labels.append(label)
        
        if not audio_file_paths:
            raise ValueError(f"No valid audio files found in {audio_dir}")
            
        return audio_file_paths, labels
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
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
    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)

        # check if audio is loaded succesfully
        if audio is None or len(audio) == 0:
            print(f"Warning: Failed to load audio from {file_path}")
            return None

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
        spectrogram_dB = (spectrogram_dB - spectrogram_dB.min()) / (
            spectrogram_dB.max() - spectrogram_dB.min() + 1e-6
        )

        return spectrogram_dB

    except Exception as e:
        print(f"Error processinf {file_path}: {str(e)}")
        return None

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
def data_set(file_paths, batch_size, target_shape, annotations_df, shuffle=True):
    """Create a TensorFlow dataset from audio files.
    
    Parameters
    ----------
    file_paths : list of str
        List of paths to audio files
    batch_size : int
        Size of batches to create
    target_shape : tuple
        Target shape for spectrograms (height, width)
    annotations_df : pandas.DataFrame
        DataFrame containing annotations for audio files
    shuffle : bool, optional
        Whether to shuffle the dataset, by default True
    """
    def process_audio_file(file_path):
        try:
            # Load and process audio using librosa
            audio, sr = librosa.load(file_path, sr=48000)  # SONYC-UST uses 48kHz
            
            # Generate mel spectrogram
            spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=target_shape[0],
                hop_length=512
            )
            
            # Convert to dB scale and normalize
            spec_db = librosa.power_to_db(spec, ref=np.max)
            spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)
            
            # Resize if needed
            if spec_norm.shape != target_shape:
                spec_norm = cv2.resize(spec_norm, (target_shape[1], target_shape[0]))
            
            # Add channel dimension
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
                # Process audio
                spec = process_audio_file(file_path)
                if spec is not None:
                    # Get label
                    label = get_label(file_path)
                    if label is not None:
                        valid_files += 1
                        yield spec, label
                    
            except Exception as e:
                print(f"Error in generator for {file_path}: {str(e)}")
                continue
        
        if valid_files == 0:
            raise ValueError("No valid audio files processed")

    # Create dataset
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
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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