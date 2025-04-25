import os
import pytest
import pandas as pd
import traceback
import utils as u
import tensorflow as tf
import tensorflow_io as tfio

def test_dataset_creation():
    """Test the dataset creation pipeline"""
    base_dir = '/Users/calodii/Desktop/stuff/home/dl4m/noise-bliss/data/sonyc-ust'
    audio_dir = os.path.join(base_dir, 'audio')
    annotations_df = pd.read_csv(os.path.join(base_dir, 'annotations.csv'))
    
    # Test parameters
    target_shape = (128, 128)
    batch_size = 32
    
    # Filter audio files by split
    train_files = annotations_df[annotations_df['split'] == 'train']['audio_filename'].unique()
    train_paths = [os.path.join(audio_dir, f) for f in train_files]
    
    print(f"\nTesting dataset creation with {len(train_paths)} files")
    print(f"First file path: {train_paths[0]}")
    
    try:
        # Create dataset
        dataset = u.data_set(
            train_paths[:100],  # Test with first 100 files
            batch_size,
            target_shape
        )
        
        # Try to get first batch
        for batch in dataset.take(1):
            x, y = batch
            print(f"\nBatch shapes:")
            print(f"Input (X): {x.shape}")
            print(f"Target (y): {y.shape}")
            assert isinstance(x, tf.Tensor), "Input is not a tensor"
            assert isinstance(y, tf.Tensor), "Target is not a tensor"
            assert x.shape[0] == batch_size, f"Unexpected batch size: {x.shape[0]}"
            assert x.shape[1:3] == target_shape, f"Unexpected shape: {x.shape[1:3]}"
            
    except Exception as e:
        pytest.fail(f"Dataset creation failed: {str(e)}\n{traceback.format_exc()}")