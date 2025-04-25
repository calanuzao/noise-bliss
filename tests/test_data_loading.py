import os
import pytest
import pandas as pd
import soundfile as sf

def test_data_directory_structure():
    base_dir = '/Users/calodii/Desktop/stuff/home/dl4m/noise-bliss/data/sonyc-ust'
    audio_dir = os.path.join(base_dir, 'audio')
    annotations_path = os.path.join(base_dir, 'annotations.csv')
    
    assert os.path.exists(base_dir), f"Base directory does not exist: {base_dir}"
    assert os.path.exists(audio_dir), f"Audio directory does not exist: {audio_dir}"
    assert os.path.exists(annotations_path), f"Annotations file does not exist: {annotations_path}"

def test_annotations_file():
    annotations_path = '/Users/calodii/Desktop/stuff/home/dl4m/noise-bliss/data/sonyc-ust/annotations.csv'
    df = pd.read_csv(annotations_path)
    
    required_columns = ['audio_filename', 'split', 'sensor_id']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    assert not df['audio_filename'].isna().any(), "Found missing audio filenames"
    assert not df['split'].isna().any(), "Found missing split values"

def test_audio_files():
    base_dir = '/Users/calodii/Desktop/stuff/home/dl4m/noise-bliss/data/sonyc-ust'
    audio_dir = os.path.join(base_dir, 'audio')
    annotations_df = pd.read_csv(os.path.join(base_dir, 'annotations.csv'))
    
    # Check first 5 audio files
    for filename in annotations_df['audio_filename'][:5]:
        file_path = os.path.join(audio_dir, filename)
        assert os.path.exists(file_path), f"Audio file missing: {file_path}"
        
        try:
            data, samplerate = sf.read(file_path)
            assert samplerate == 48000, f"Unexpected sample rate {samplerate} for {filename}"
            assert len(data.shape) <= 2, f"Unexpected audio channels for {filename}"
        except Exception as e:
            pytest.fail(f"Failed to read audio file {filename}: {str(e)}")

def test_split_distribution():
    annotations_path = '/Users/calodii/Desktop/stuff/home/dl4m/noise-bliss/data/sonyc-ust/annotations.csv'
    df = pd.read_csv(annotations_path)
    
    splits = df['split'].value_counts()
    assert 'train' in splits.index, "No training data found"
    assert 'validate' in splits.index, "No validation data found"
    assert 'test' in splits.index, "No test data found"

def test_audio_file_paths():
    """Test if audio file paths are correctly constructed"""
    base_dir = '/Users/calodii/Desktop/stuff/home/dl4m/noise-bliss/data/sonyc-ust'
    audio_dir = os.path.join(base_dir, 'audio')
    annotations_df = pd.read_csv(os.path.join(base_dir, 'annotations.csv'))
    
    print(f"\nTotal audio files in annotations: {len(annotations_df)}")
    
    # Test first few files from each split
    for split in ['train', 'validate', 'test']:
        split_files = annotations_df[annotations_df['split'] == split]['audio_filename'].iloc[:3]
        print(f"\nTesting {split} files:")
        for filename in split_files:
            file_path = os.path.join(audio_dir, filename)
            exists = os.path.exists(file_path)
            print(f"- {filename}: {'✓' if exists else '✗'}")
            assert exists, f"File not found: {file_path}"

def test_audio_loading():
    """Test if audio files can be loaded and processed"""
    base_dir = '/Users/calodii/Desktop/stuff/home/dl4m/noise-bliss/data/sonyc-ust'
    audio_dir = os.path.join(base_dir, 'audio')
    annotations_df = pd.read_csv(os.path.join(base_dir, 'annotations.csv'))
    
    # Test one file from each split
    for split in ['train', 'validate', 'test']:
        filename = annotations_df[annotations_df['split'] == split]['audio_filename'].iloc[0]
        file_path = os.path.join(audio_dir, filename)
        
        print(f"\nTesting {split} file: {filename}")
        try:
            data, sr = sf.read(file_path)
            print(f"- Sample rate: {sr}")
            print(f"- Shape: {data.shape}")
            print(f"- Duration: {len(data)/sr:.2f}s")
        except Exception as e:
            pytest.fail(f"Failed to load {filename}: {str(e)}")