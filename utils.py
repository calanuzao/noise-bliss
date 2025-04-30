"""

This utils.py includes:

All necessary imports
Label taxonomy definition
Data loading function that works with the wrapper
Audio processing functions
Dataset creation with TensorFlow
Data augmentation
Visualization utilities
The main changes are:

Removed the SonycUSTDataset class (now in wrapper.py)
Updated load_data() to work with the wrapper
Streamlined the data processing pipeline
Added visualization functions for training monitoring

"""
#========================================================================================
#                                   IMPORTS
#========================================================================================
# Core libraries
import os
import random
import datetime
from pathlib import Path

# Data handling
import numpy as np
import pandas as pd

# Audio processing (librosa)
import librosa
import librosa.display

# Visualization tools
import matplotlib.pyplot as plt
import cv2

# TensorFlow and Keras
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from keras import layers

# Metrics and evaluation
from sklearn.metrics import classification_report

# Utilities
from tqdm import tqdm

# Numerical stability constant
eps = np.finfo(float).eps

# Define the 8 coarse-grained label columns
COARSE_PRESENCE_COLS = [
    '1_engine_presence',
    '2_machinery-impact_presence',
    '3_non-machinery-impact_presence',
    '4_powered-saw_presence',
    '5_alert-signal_presence',
    '6_music_presence',
    '7_human-voice_presence',
    '8_dog_presence'
]

# Taxonomy mapping for SONYC-UST dataset
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

#========================================================================================
#                                   DATA LOADING
#========================================================================================

def load_data(dataset):
    """
    Load SONYC-UST dataset using the custom wrapper.

    This function resolves valid audio file paths and constructs 
    multi-hot label vectors based on presence annotations.

    Parameters
    ----------
    dataset : SonycUST
        An instance of the SONYC-UST dataset wrapper.

    Returns
    -------
    audio_file_paths : list of str
        Paths to audio files with corresponding annotations.
    labels : list of list of int
        Multi-hot encoded label vectors based on class presence.
    """
    try:
        # Load the full annotation DataFrame
        annotations_df = dataset.load_annotations()

        audio_file_paths = []
        labels = []

        # Get unique filenames from annotations
        unique_files = annotations_df['audio_filename'].unique()

        for audio_file in unique_files:
            file_path = os.path.join(dataset.data_dir, 'audio', audio_file)

            # Only proceed if the audio file exists
            if os.path.exists(file_path):
                file_annotations = annotations_df[annotations_df['audio_filename'] == audio_file]
                label = [0] * len(label_taxonomy)

                # Check each category in the taxonomy for presence
                for category, idx in label_taxonomy.items():
                    column = f"{idx+1}_{category}_presence"
                    if column in file_annotations.columns and (file_annotations[column] == 1).any():
                        label[idx] = 1

                audio_file_paths.append(file_path)
                labels.append(label)

        # Ensure some valid data was collected
        if not audio_file_paths:
            raise ValueError("No valid audio files found.")

        return audio_file_paths, labels

    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
#========================================================================================
#                            AUDIO PROCESSING UTILITIES
#========================================================================================

def process_audio(file_path_tensor, target_shape=(128, 128)):
    """
    Load an audio file and convert it into a normalized mel spectrogram with a fixed shape.

    Parameters
    ----------
    file_path_tensor : tf.Tensor or str
        Path to the audio file, typically passed as a TensorFlow string tensor.
    target_shape : tuple of int, default=(128, 128)
        Desired output spectrogram shape (mel bands, time frames).

    Returns
    -------
    np.ndarray
        Normalized mel spectrogram of shape (target_shape[0], target_shape[1], 1).
        If the file is missing or an error occurs, returns a zero-filled array.
    """
    try:
        # Convert TensorFlow string tensor to a Python string
        if isinstance(file_path_tensor, tf.Tensor):
            file_path = file_path_tensor.numpy().decode('utf-8')
        else:
            file_path = str(file_path_tensor)

        # Ensure the file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return np.zeros((target_shape[0], target_shape[1], 1), dtype=np.float32)

        # Load 10-second audio clip at 22.05 kHz
        audio, sr = librosa.load(file_path, sr=22050, duration=10)

        # Skip if audio is empty or too short
        if len(audio) < sr * 0.5:
            print(f"Audio file too short: {file_path}")
            return np.zeros((target_shape[0], target_shape[1], 1), dtype=np.float32)

        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=target_shape[0],
            n_fft=2048,
            hop_length=512
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)

        # Ensure output width matches target shape
        if mel_spec_norm.shape[1] != target_shape[1]:
            mel_spec_resized = np.zeros((target_shape[0], target_shape[1]))
            width = min(mel_spec_norm.shape[1], target_shape[1])
            mel_spec_resized[:, :width] = mel_spec_norm[:, :width]
            mel_spec_norm = mel_spec_resized

        # Add channel dimension for CNNs
        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)

        # Final shape validation
        assert mel_spec_norm.shape == (target_shape[0], target_shape[1], 1), f"Wrong shape: {mel_spec_norm.shape}"

        return mel_spec_norm.astype(np.float32)

    except Exception as e:
        print(f"Error processing {file_path_tensor}: {str(e)}")
        return np.zeros((target_shape[0], target_shape[1], 1), dtype=np.float32)

def augment_spectrogram(spec):
    """
    Apply simple random augmentations to a mel spectrogram.

    This includes random frequency and time shifts using tensor rolling.
    Used for data augmentation to improve model generalization.

    Parameters
    ----------
    spec : tf.Tensor
        Input mel spectrogram tensor of shape (freq_bins, time_steps, 1) or similar.

    Returns
    -------
    tf.Tensor
        Augmented spectrogram with the same shape.
    """
    # Apply random shift along frequency axis (vertical roll)
    freq_shift = tf.random.uniform([], minval=0, maxval=20, dtype=tf.int32)
    spec = tf.roll(spec, shift=freq_shift, axis=0)

    # Apply random shift along time axis (horizontal roll)
    time_shift = tf.random.uniform([], minval=0, maxval=20, dtype=tf.int32)
    spec = tf.roll(spec, shift=time_shift, axis=1)

    return spec

#========================================================================================
#                               DATASET CREATION
#========================================================================================

def compute_class_weights(annotations, label_columns):
    """
    Compute class weights to address label imbalance in multi-label classification.

    This method assigns higher weights to less frequent classes using inverse-frequency
    scaling, then normalizes the weights so their mean is 1. These weights can be used
    to rescale loss contributions or drive data augmentation.

    Parameters
    ----------
    annotations : pd.DataFrame
        DataFrame containing binary multi-hot label columns for each audio file.
    label_columns : list of str
        List of column names corresponding to the class presence indicators.

    Returns
    -------
    list of float
        Normalized class weights (mean = 1), ordered to match `label_columns`.
    """

    # Count total positives per class
    label_sums = annotations[label_columns].sum().values

    # Total number of samples in dataset
    total = len(annotations)

    # Inverse frequency weighting (add small epsilon to avoid division by 0)
    weights = total / (label_sums + 1e-6)

    # Normalize weights so the average is 1
    weights /= np.mean(weights)

    return weights.tolist()

def prepare_data(annotations, dataset, target_shape=(128, 128), max_samples=10000):
    """
    Prepare batched TensorFlow datasets for audio classification using spectrograms.

    This function reads annotations and audio paths, verifies file existence,
    computes spectrograms, and returns tf.data.Datasets for training and validation.

    Parameters
    ----------
    annotations : pd.DataFrame
        DataFrame containing metadata and multi-hot label annotations for each audio sample.
    dataset : object
        A dataset wrapper that exposes get_audio_path(filename) to resolve full audio paths.
    target_shape : tuple, default=(128, 128)
        Target shape of output spectrograms (height, width).
    max_samples : int, default=10000
        Maximum number of training and validation samples to include.

    Returns
    -------
    tuple of tf.data.Dataset
        (train_dataset, val_dataset) with spectrogram-label pairs.
    """

    # Filter training and validation subsets
    train_df = annotations[annotations['split'] == 'train'].head(max_samples)
    val_df = annotations[annotations['split'] == 'validate'].head(max_samples)

    print(f"Using {len(train_df)} training samples and {len(val_df)} validation samples")

    def create_tf_dataset(df, is_training=True):
        """
        Internal helper to process a subset of annotations into a tf.data.Dataset.
        """
        audio_files = []
        labels = []

        # Verify file paths and collect valid label pairs
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking audio paths"):
            audio_path = dataset.get_audio_path(row['audio_filename'])

            if audio_path is None:
                print(f"File not found via get_audio_path: {row['audio_filename']}")
                continue
            if not os.path.exists(audio_path):
                print(f"Path returned but file missing: {audio_path}")
                continue

            label = [row[col] for col in COARSE_PRESENCE_COLS]
            audio_files.append(str(audio_path))
            labels.append(label)

        print(f"Found {len(audio_files)} valid audio files")

        # Convert paths and labels to NumPy arrays
        audio_files = np.array(audio_files)
        labels = np.array(labels, dtype=np.float32)

        # Generate spectrograms and filter invalid/mismatched shapes
        processed_specs = []
        processed_labels = []

        for audio_file, label in tqdm(zip(audio_files, labels), total=len(audio_files), desc="Processing spectrograms"):
            try:
                spec = process_audio(audio_file, target_shape)
                if spec is not None and spec.shape == (target_shape[0], target_shape[1], 1):
                    processed_specs.append(spec)
                    processed_labels.append(label)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

        # Stack into tensors or create empty fallback tensors
        if processed_specs:
            processed_specs = np.stack(processed_specs)
            processed_labels = np.stack(processed_labels)
            ds = tf.data.Dataset.from_tensor_slices((processed_specs, processed_labels))
        else:
            ds = tf.data.Dataset.from_tensor_slices((
                np.zeros((0, target_shape[0], target_shape[1], 1), dtype=np.float32),
                np.zeros((0, len(COARSE_PRESENCE_COLS)), dtype=np.float32)
            ))

        # Apply batching, shuffling, prefetching
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        if is_training:
            ds = ds.shuffle(1000)

        return ds

    # Construct training and validation datasets
    train_data = create_tf_dataset(train_df, is_training=True)
    val_data = create_tf_dataset(val_df, is_training=False)

    return train_data, val_data

def data_set(file_paths, batch_size, target_shape, annotations_df, shuffle=True):
    """
    Create a TensorFlow dataset from a list of audio file paths.

    This function loads audio files, computes mel spectrograms, normalizes them,
    applies optional augmentation, and returns a batched tf.data.Dataset of
    (spectrogram, label) pairs suitable for training.

    Parameters
    ----------
    file_paths : list of str
        List of full paths to audio files.
    batch_size : int
        Number of samples per batch.
    target_shape : tuple
        Target shape (height, width) of output spectrograms.
    annotations_df : pd.DataFrame
        Annotation dataframe containing binary presence labels.
    shuffle : bool, default=True
        Whether to shuffle the dataset during loading.

    Returns
    -------
    tf.data.Dataset
        A batched and prefetched dataset of (spectrogram, label) pairs.
    """

    def process_audio_file(file_path):
        """
        Load audio and compute normalized mel spectrogram.

        Returns
        -------
        np.ndarray or None
            Normalized mel spectrogram or None if processing fails.
        """
        try:
            audio, sr = librosa.load(file_path, sr=48000)
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
        """
        Extract multi-hot label vector from annotations DataFrame.

        Returns
        -------
        np.ndarray or None
            Label vector for the given audio path.
        """
        try:
            filename = os.path.basename(audio_path)
            file_annotations = annotations_df[annotations_df['audio_filename'] == filename]

            if file_annotations.empty:
                print(f"No annotations found for {filename}")
                return None

            label = [0] * len(label_taxonomy)
            for category, idx in label_taxonomy.items():
                column = f"{idx + 1}_{category}_presence"
                if column in file_annotations.columns and (file_annotations[column] == 1).any():
                    label[idx] = 1

            return np.array(label, dtype=np.float32)

        except Exception as e:
            print(f"Error getting label for {audio_path}: {str(e)}")
            return None

    def generator():
        """
        Generator function yielding (spectrogram, label) pairs.
        Applies augmentation if enabled.
        """
        valid_files = 0
        for file_path in file_paths:
            try:
                spec = process_audio_file(file_path)
                if spec is not None:
                    label = get_label(file_path)
                    if label is not None:
                        valid_files += 1
                        if shuffle:
                            spec = augment_spectrogram(spec)
                        yield spec, label
            except Exception as e:
                print(f"Error in generator for {file_path}: {str(e)}")
                continue

        if valid_files == 0:
            raise ValueError("No valid audio files processed")

    # Define output shapes and types for tf.data.Dataset
    output_signature = (
        tf.TensorSpec(shape=(target_shape[0], target_shape[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(len(label_taxonomy),), dtype=tf.float32)
    )

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # Optional shuffling
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def analyze_dataset(annotations):
    """
    Analyze dataset characteristics including split balance, class label frequency,
    and temporal distribution across hours of the day.

    Parameters
    ----------
    annotations : pd.DataFrame
        DataFrame containing metadata and multi-label presence indicators for each sample.
    """
    print("Dataset Analysis Summary")

    # Split distribution (train, validate, test)
    print("\nSplit Distribution:")
    print(annotations['split'].value_counts())

    # Identify all coarse-grained binary label columns
    coarse_labels = [
        col for col in annotations.columns
        if col.endswith('_presence') and len(col.split('_')) == 2
    ]

    # Display total presence for each class label
    print("\nCoarse Label Distribution:")
    total_samples = len(annotations)
    for label in coarse_labels:
        present_count = annotations[label].sum()
        percentage = (present_count / total_samples) * 100
        print(f"{label}: {present_count} ({percentage:.2f}%)")

    # Hourly distribution of recording timestamps
    print("\nHourly Distribution of Recordings:")
    print(annotations['hour'].value_counts().sort_index())

#========================================================================================
#                               TEMPORAL ENCODING
#========================================================================================

def filter_existing_files(audio_paths, temporal_features, labels):
    """
    Filter out audio files that are missing on disk.

    This function ensures alignment between valid audio paths, associated temporal
    features, and labels

    Parameters
    ----------
    audio_paths : list of str
        List of absolute or relative audio file paths.
    temporal_features : np.ndarray
        Numpy array of corresponding temporal metadata features.
    labels : np.ndarray
        Numpy array of corresponding multi-label binary vectors.

    Returns
    -------
    tuple of np.ndarray
        (filtered_audio_paths, filtered_temporal_features, filtered_labels)
    """
    filtered_paths = []
    filtered_temporal = []
    filtered_labels = []

    for path, temp, label in zip(audio_paths, temporal_features, labels):
        if os.path.exists(path):
            filtered_paths.append(path)
            filtered_temporal.append(temp)
            filtered_labels.append(label)
        else:
            print(f"[Skipping] Missing file: {path}")

    return (
        np.array(filtered_paths),
        np.array(filtered_temporal),
        np.array(filtered_labels)
    )

def encode_temporal_features(df):
    """
    Encode time metadata into a fixed 9-dimensional vector.

    This includes normalized hour and week of year, and one-hot encoding
    of the day of the week (0 = Monday, 6 = Sunday)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'hour', 'week', and 'day' columns.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 9), where each row is a temporal feature vector.
    """
    hour_norm = df['hour'] / 23.0
    week_norm = df['week'] / 52.0

    all_days = pd.get_dummies(pd.Series(range(7)), prefix='day').columns
    day_onehot = pd.get_dummies(df['day'], prefix='day').reindex(columns=all_days, fill_value=0)

    return np.concatenate([
        hour_norm.values[:, np.newaxis],
        week_norm.values[:, np.newaxis],
        day_onehot.values
    ], axis=1)

#========================================================================================
#                               YAMNET EMBEDDING
#========================================================================================
def map_with_yamnet(file_path, label):
    """
    Map function that extracts YAMNet embeddings from an audio file.
    
    Parameters:
        file_path (tf.Tensor): Tensor containing the file path string
        label (tf.Tensor): Corresponding multi-label vector
        
    Returns:
        (tf.Tensor, tf.Tensor): (YAMNet embedding, label)
    """
    def load_and_embed(path):
        import tensorflow_hub as hub
        import librosa
        import numpy as np
        import tensorflow as tf

        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        path = path.numpy().decode("utf-8")
        audio, _ = librosa.load(path, sr=16000)
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(audio_tensor)
        return embeddings.numpy().astype(np.float32)

    embedding = tf.py_function(
        func=load_and_embed,
        inp=[file_path],
        Tout=tf.float32
    )
    embedding.set_shape([None, 1024])  # Set shape if known
    return embedding, label

#========================================================================================
#                                   PLOTTING
#========================================================================================

def visualize_distributions(annotations):
    """
    Plot basic dataset statistics from the annotation metadata.

    This includes:
    - Distribution of samples across train/validate/test splits.
    - Distribution of recordings across hours of the day.

    Parameters
    ----------
    annotations : pd.DataFrame
        DataFrame containing metadata, including 'split' and 'hour' columns.
    """

    # Create a figure with two subplots
    plt.figure(figsize=(15, 10))

    # Plot 1: Distribution of samples by dataset split
    plt.subplot(2, 1, 1)
    splits = annotations['split'].value_counts()
    plt.bar(splits.index, splits.values)
    plt.title("Dataset Split Distribution")
    plt.ylabel("Number of Samples")

    # Plot 2: Distribution of recordings by hour of day
    plt.subplot(2, 1, 2)
    hours = annotations['hour'].value_counts().sort_index()
    plt.plot(hours.index, hours.values, marker='o')
    plt.title("Hourly Distribution of Recordings")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Recordings")
    plt.grid(True)

    # Final layout adjustment
    plt.tight_layout()
    plt.show()

def plot_class_imbalance(annotations_df, label_columns):
    """
    Plot class imbalance for multi-label classification datasets.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        DataFrame containing binary presence columns for each class.
    label_columns : list of str
        List of column names corresponding to presence labels.
    """
    # Convert presence columns to numeric (handle any unexpected values safely)
    presence_clean = annotations_df[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Clip values to [0, 1] to ensure binary validity
    presence_clipped = presence_clean.clip(lower=0, upper=1)

    # Compute total positive samples per class
    class_counts = presence_clipped.sum().astype(int)

    # Create bar plot
    plt.figure(figsize=(16, 6))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title("Class Imbalance in Dataset")
    plt.ylabel("Number of Positive Samples")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio_path):
    """
    Plot the mel spectrogram of a given audio file.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file.
    """
    # Load audio file at a fixed sample rate
    audio, sr = librosa.load(audio_path, sr=22050)

    # Compute mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Convert to dB scale for visualization
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot the mel spectrogram
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def plot_cnn_tcn_training_curves(cnn_history, tcn_history):
    """
    Plot loss and binary accuracy curves for CNN and TCN models.

    Parameters
    ----------
    cnn_history : keras.callbacks.History
        Training history from CNN model.
    tcn_history : keras.callbacks.History
        Training history from TCN model.
    """
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['loss'], label='CNN Train')
    plt.plot(cnn_history.history['val_loss'], label='CNN Val')
    plt.plot(tcn_history.history['loss'], label='TCN Train')
    plt.plot(tcn_history.history['val_loss'], label='TCN Val')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['binary_accuracy'], label='CNN Train')
    plt.plot(cnn_history.history['val_binary_accuracy'], label='CNN Val')
    plt.plot(tcn_history.history['binary_accuracy'], label='TCN Train')
    plt.plot(tcn_history.history['val_binary_accuracy'], label='TCN Val')
    plt.title('Binary Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_augmented_model_history(history):
    """
    Plot training and validation loss and binary accuracy for an augmented model.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by model.fit().
    """
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 2: Binary Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'], label='Training Binary Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Binary Accuracy')
    plt.title('Binary Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Visualize training and validation loss and accuracy over epochs.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by model.fit().
    """
    plt.figure(figsize=(12, 4))

    # Plot loss over epochs
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.grid(True)

    # Plot accuracy over epochs
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('binary_accuracy', history.history.get('accuracy')), label='Training Accuracy')
    plt.plot(history.history.get('val_binary_accuracy', history.history.get('val_accuracy')), label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_training_metrics_yamnet(history):
    """
    Plot training loss and accuracy from model.fit() history object.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history returned by model.fit().
    """
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    if 'val_binary_accuracy' in history.history:
        plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.title('Binary Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_training_metrics_yamnet(history):
    """
    Plot training and validation metrics from a YAMNet-based model training history.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by model.fit().
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
    plt.title('YAMNet Model: Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_binary_accuracy' in history.history:
        plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy', linestyle='--', linewidth=2)
    plt.title('YAMNet Model: Binary Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()