import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import os
import torch
import torchaudio


# ---------- BOROUGH CODE TO NAME MAPPING ----------
borough_map = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island"
}

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
    Loads all .wav files and infers labels from parent folder names.

    Args:
        data_path (str or Path): Root directory where audio files are stored.

    Returns:
        audio_files (list of str): List of full audio file paths.
        labels (list of str): Corresponding label for each audio file.
    """
    data_path = Path(data_path)
    audio_files = list(data_path.rglob("*.wav"))
    labels = [f.parent.name for f in audio_files]
    
    # Convert Path objects to strings for compatibility
    audio_files = [str(f) for f in audio_files]

    return audio_files, labels



# ---------- Construct datetime ----------
def construct_datetime(df):
    """
    Constructs a 'datetime' column from 'year', 'week', 'day', 'hour'.

    Returns the modified DataFrame.
    """
    # applly the datetime computation to each row using the "apply" method
    dt_series = df.apply(lambda row: datetime.datetime.strptime(
        # format the year, week, and day values; week and day incremented by 1
        f"{int(row['year'])} {int(row['week']) + 1} {int(row['day']) + 1}",
        "%G %V %u"                                      # G: century; V: Week; U: Weekday
    ) + pd.Timedelta(hours=int(row['hour'])), axis=1)   # add the hour information using pd.Timedelta
    
    df['datetime'] = pd.to_datetime(dt_series)          # convert the series to datetime
    return df                                           # return the dataframe

# ---------- Construct location_id ----------
def construct_location_id(df):
    """
    Constructs a unique 'location_id' column from 'sensor_id', 'borough', and 'block'.
    
    Returns the modified DataFrame.
    """
    df['location_id'] = (          
        df['borough'].astype(str) + "-" +   
        df['block'].astype(str) + "-" +
        df['sensor_id'].astype(str)
    )
    return df

# ---------- Compute and plot location-based distributions ----------
def compute_and_plot_location_distributions(df):
    """
    Plots sensor-based and borough-based distributions.
    Assumes 'location_id' and 'borough' columns exist.
    """
    # Count by sensor
    sensor_counts = df['sensor_id'].value_counts().sort_index()

    # Count by borough
    borough_counts = df['borough'].value_counts().sort_index()

    # Count by full location_id
    location_counts = df['location_id'].value_counts().sort_index()

    plot_distribution(sensor_counts, "Sensor ID", "Recordings", "Recordings per Sensor")
    plot_distribution(borough_counts, "Borough Code", "Recordings", "Recordings per Borough")
    plot_distribution(location_counts, "Location ID", "Recordings", "Recordings per Unique Location", kind='bar')

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


# ---------- PLOT BY SPLITS ----------
import matplotlib.pyplot as plt

def plot_distributions_by_split(df):
    """
    Plots hourly, daily, weekly, monthly, and yearly distributions by split
    using colorblind-safe colors and enhanced visibility.
    """
    # Define high-contrast, colorblind-safe colors
    split_colors = {
        'train': '#E69F00',     # orange
        'validate': '#56B4E9',  # sky blue
        'test': '#009E73'       # green
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