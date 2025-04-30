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

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import keras
from tensorflow import keras 
from keras import layers
import random
import librosa
import librosa.display
import cv2
import os
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm

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

def load_data(dataset):
    """
    Load SONYC-UST dataset using the wrapper.

    Parameters
    ----------
    dataset : SonycUST
        Instance of the SonycUST dataset wrapper

    Returns
    -------
    audio_file_paths : list of str
        List of paths to audio files
    labels : list of list
        List of multi-hot encoded labels for each audio file
    """
    try:
        # Load annotations using the wrapper
        annotations_df = dataset.load_annotations()
        
        audio_file_paths = []
        labels = []
        
        # Process unique audio files
        unique_files = annotations_df['audio_filename'].unique()
        
        for audio_file in unique_files:
            # Use wrapper to get audio path
            file_path = os.path.join(dataset.data_dir, 'audio', audio_file)
            
            if os.path.exists(file_path):
                # Get all annotations for this file
                file_annotations = annotations_df[annotations_df['audio_filename'] == audio_file]
                
                # Create multi-hot encoded label
                label = [0] * len(label_taxonomy)
                
                # For each category in taxonomy
                for category, idx in label_taxonomy.items():
                    column = f"{idx+1}_{category}_presence"
                    if column in file_annotations.columns and (file_annotations[column] == 1).any():
                        label[idx] = 1
                
                audio_file_paths.append(file_path)
                labels.append(label)
        
        if not audio_file_paths:
            raise ValueError(f"No valid audio files found")
            
        return audio_file_paths, labels
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def process_audio(file_path_tensor, target_shape=(128, 128)):
    """Process audio file into spectrogram with target shape"""
    try:
        # Convert tensor to string
        if isinstance(file_path_tensor, tf.Tensor):
            file_path = file_path_tensor.numpy().decode('utf-8')
        else:
            file_path = str(file_path_tensor)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            # Return empty spectrogram with correct shape
            return np.zeros((target_shape[0], target_shape[1], 1), dtype=np.float32)
        
        # Load audio file
        audio, sr = librosa.load(file_path, sr=22050, duration=10)
        
        # Check if audio is empty or too short
        if len(audio) < sr * 0.5:  # Less than 0.5 seconds
            print(f"Audio file too short: {file_path}")
            return np.zeros((target_shape[0], target_shape[1], 1), dtype=np.float32)
            
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=target_shape[0],  # Set number of mel bands to match target height
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        
        # Ensure correct shape - explicitly resize if needed
        if mel_spec_norm.shape[1] != target_shape[1]:
            # Use resize to get exact dimensions
            mel_spec_resized = np.zeros((target_shape[0], target_shape[1]))
            # Copy available data, padding or truncating as needed
            width = min(mel_spec_norm.shape[1], target_shape[1])
            mel_spec_resized[:, :width] = mel_spec_norm[:, :width]
            mel_spec_norm = mel_spec_resized
            
        # Add channel dimension for CNN (height, width, channels)
        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)
        
        # Final shape check
        assert mel_spec_norm.shape == (target_shape[0], target_shape[1], 1), f"Wrong shape: {mel_spec_norm.shape}"
            
        return mel_spec_norm.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {file_path_tensor}: {str(e)}")
        # Return empty spectrogram with correct shape
        return np.zeros((target_shape[0], target_shape[1], 1), dtype=np.float32)

def data_set(file_paths, batch_size, target_shape, annotations_df, shuffle=True):
    """
    Create a TensorFlow dataset from audio files.
    
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
                        if shuffle:
                            spec = augment_spectrogram(spec)
                        yield spec, label 
                    
            except Exception as e:
                print(f"Error in generator for {file_path}: {str(e)}")
                continue
        
        if valid_files == 0:
            raise ValueError("No valid audio files processed")

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

def augment_spectrogram(spec):
    """
    Apply random augmentations to mel spectrogram.
    """
    freq_mask = tf.random.uniform([], 0, 20, dtype=tf.int32)
    spec = tf.roll(spec, freq_mask, axis=0)

    time_mask = tf.random.uniform([], 0, 20, dtype=tf.int32)
    spec = tf.roll(spec, time_mask, axis=1)

    return spec

def plot_spectrogram(audio_path):
    """
    Plot mel spectrogram for a given audio file.
    """
    audio, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_spec_db, 
        y_axis='mel', 
        x_axis='time',
        sr=sr
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot training history metrics.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


""" DATASET DEFINITIONS """

def analyze_dataset(annotations):
    """
    Analyze general dataset characteristics, including:
    - Split distribution
    - Coarse label frequency
    - Temporal (hourly) distribution

    Parameters:
    ----------
    annotations : pd.DataFrame
        DataFrame containing metadata and presence labels for each audio sample.
    """
    print("Dataset Analysis:")
    
    # --- Dataset split counts ---
    print("\nSplit Distribution:")
    print(annotations['split'].value_counts())
    
    # --- Coarse label presence counts ---
    coarse_labels = [
        col for col in annotations.columns 
        if col.endswith('_presence') and len(col.split('_')) == 2
    ]
    
    print("\nCoarse Label Distribution:")
    total = len(annotations)
    for label in coarse_labels:
        count = annotations[label].sum()
        print(f"{label}: {count} ({(count / total) * 100:.2f}%)")
    
    # --- Hour-of-day distribution ---
    print("\nTemporal Distribution:")
    print("Hours distribution:")
    print(annotations['hour'].value_counts().sort_index())


def visualize_distributions(annotations):
    """
    Visualize:
    - Dataset split proportions
    - Recording distribution by hour of day

    Parameters:
    ----------
    annotations : pd.DataFrame
        DataFrame with dataset metadata and labels.
    """
    plt.figure(figsize=(15, 10))  # Layout for 2 subplots

    # --- Subplot 1: Split distribution ---
    plt.subplot(2, 1, 1)
    splits = annotations['split'].value_counts()
    plt.bar(splits.index, splits.values)
    plt.title('Dataset Split Distribution')
    plt.ylabel('Number of Samples')

    # --- Subplot 2: Hourly activity ---
    plt.subplot(2, 1, 2)
    hours = annotations['hour'].value_counts().sort_index()
    plt.plot(hours.index, hours.values, marker='o')
    plt.title('Hourly Distribution of Recordings')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Recordings')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # --- Compute Class Weights to Handle Imbalance ---
def compute_class_weights(annotations, label_columns):
    """
    Compute inverse-frequency class weights based on label prevalence.
    """
    label_sums = annotations[label_columns].sum().values
    total = len(annotations)
    weights = total / (label_sums + 1e-6)  # Avoid division by zero
    weights /= np.mean(weights)  # Normalize so mean = 1
    return weights.tolist()


def prepare_data(annotations, dataset, target_shape=(128, 128), max_samples=10000):
    """
    Prepare TensorFlow datasets for training and validation.
    """
    # Split annotations into train and validate subsets
    train_df = annotations[annotations['split'] == 'train'].head(max_samples)
    val_df = annotations[annotations['split'] == 'validate'].head(max_samples)

    print(f"Using {len(train_df)} training samples and {len(val_df)} validation samples")

    def create_tf_dataset(df, is_training=True):
        """
        Build TensorFlow dataset from file paths and multi-hot labels.
        """
        audio_files, labels = [], []

        # --- Step 1: Verify all audio file paths and labels ---
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking audio paths"):
            audio_path = dataset.get_audio_path(row['audio_filename'])
            if audio_path is None:
                print(f"File not found via get_audio_path: {row['audio_filename']}")
            elif not os.path.exists(audio_path):
                print(f"Path returned but file missing: {audio_path}")
            else:
                label = [row[col] for col in COARSE_PRESENCE_COLS]
                audio_files.append(str(audio_path))
                labels.append(label)

        print(f"Found {len(audio_files)} valid audio files")

        # --- Step 2: Convert to NumPy arrays ---
        audio_files = np.array(audio_files)
        labels = np.array(labels, dtype=np.float32)

        # --- Step 3: Process spectrograms from audio files ---
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

        # --- Step 4: Create tf.data.Dataset ---
        if processed_specs:
            processed_specs = np.stack(processed_specs)
            processed_labels = np.stack(processed_labels)
            ds = tf.data.Dataset.from_tensor_slices((processed_specs, processed_labels))
        else:
            ds = tf.data.Dataset.from_tensor_slices((
                np.zeros((0, target_shape[0], target_shape[1], 1), dtype=np.float32),
                np.zeros((0, len(COARSE_PRESENCE_COLS)), dtype=np.float32)
            ))

        # --- Step 5: Batch, Prefetch, Shuffle ---
        ds = ds.batch(32)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        if is_training:
            ds = ds.shuffle(1000)

        return ds

    # --- Create training and validation datasets ---
    train_data = create_tf_dataset(train_df, is_training=True)
    val_data = create_tf_dataset(val_df, is_training=False)

    return train_data, val_data