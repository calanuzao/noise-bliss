import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torchaudio

def load_metadata(csv_path):
    """Loads metadata from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the metadata.
    """
    return pd.read_csv(csv_path)

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
                    label = label.decode('utf8')  # Fixed this line to decode label
                labels.append(label)

    return np.array(audio_files), np.array(labels)

def plot_waveform(waveform, sample_rate):
    """Plots the waveform of an audio signal.

    Args:
        waveform (np.ndarray): The audio waveform.
        sample_rate (int): The sample rate of the audio signal.
    """
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def compute_mel_spectrogram(audio, sample_rate=22050, n_mels=128, hop_length=512):
    """
    Compute the normalized Mel spectrogram of an audio signal.

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
    # Hint: use librosa melspectrogram and librosa power_to_db
    # YOUR CODE HERE
    # perceptually motivated spectrogram that mirrors human perception of spectral components
    audio = librosa.to_mono(audio)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,                # array of audio data
        sr=sample_rate,         # set the sample rate
        hop_length=hop_length,  # determines temporal resolution
        n_mels=n_mels           # determines spectral resolution
    )
    # convert to a logarithmic scale in order to normalize the peak value
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

# ---------- Construct datetime ----------
def construct_datetime(df):
    """
    Constructs a 'datetime' column from 'year', 'week', 'day', 'hour'.

    Returns the modified DataFrame.
    """
    dt_series = df.apply(lambda row: datetime.datetime.strptime(
        f"{int(row['year'])} {int(row['week']) + 1} {int(row['day']) + 1}",
        "%G %V %u"
    ) + pd.Timedelta(hours=int(row['hour'])), axis=1)

    df['datetime'] = pd.to_datetime(dt_series)
    return df

# ---------- Single distribution plot ----------
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

# ---------- Compute and plot all distributions ----------
def compute_and_plot_distributions(df):
    """
    Plots hourly, daily, weekly, monthly, yearly distributions.
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

# ---------- PLOT BY SPLITS ----------
def plot_distributions_by_split(df):
    """
    Plots hourly, daily, weekly, monthly, and yearly distributions by split.
    """
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
            plt.plot(counts.index, counts.values, label=split.capitalize())

        plt.title(f"{name} Distribution by Split")
        plt.xlabel(name)
        plt.ylabel("Number of Recordings")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
  

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
    # add more processing steps

    return audio