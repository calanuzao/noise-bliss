# Noise Bliss: SONYC-UST Audio Event Detection Project
Denoising Algorithms for Acoustic Scene Classification in Deep Learning Models

## Project Overview
This project focuses on urban sound event detection using deep learning techniques (CNN and TCN architectures) applied to the SONYC-UST dataset. We've built a custom Soundata wrapper for the dataset and implemented a complete pipeline for audio processing, model training, and evaluation.

## Key Components

### 1. Dataset Integration (`wrapper.py`)
We created a custom Soundata wrapper for the SONYC-UST dataset:
- Implemented `SonycUST` class extending Soundata's base functionality
- Added methods for loading annotations, accessing audio files, and validating dataset structure
- Integrated with SONYC's 8-class coarse-grained taxonomy

### 2. Audio Processing Pipeline (`utils.py`)
Enhanced the utilities module with:
- Audio loading and processing functions
- Spectrogram generation with consistent shapes
- Data augmentation techniques
- TensorFlow dataset creation
- Visualization utilities for spectrograms and training metrics

### 3. Model Architecture (`classification.py`)
Implemented two different model architectures:
- **CNN Model**: Convolutional Neural Network optimized for spectrogram input
- **TCN Model**: Temporal Convolutional Network for handling time-series aspects
- Added model compilation and training functions with appropriate metrics

### 4. Training Pipeline (`pipeline.ipynb`)
Created a comprehensive notebook that:
- Validates the dataset structure and integrity
- Analyzes label distribution and temporal patterns
- Processes audio files into spectrograms
- Creates TensorFlow datasets for training and validation
- Trains both CNN and TCN models
- Visualizes training progress and model performance

## Data Insights
- The SONYC-UST dataset contains 62,022 annotations
- 8 coarse-grained sound categories
- Rich metadata including location, time, and annotator information
- Dataset is split into train/validation/test sets
- Audio recordings have varying temporal patterns throughout the day

## Further Work

### Immediate Tasks
1. **Dataset Extension**:
   - Complete the Soundata wrapper implementation
   - Add support for fine-grained label taxonomy

2. **Model Improvements**:
   - Optimize hyperparameters for both CNN and TCN models
   - Implement more advanced architectures (e.g., ResNet, Attention mechanisms)
   - Add model checkpointing and early stopping

3. **Data Processing**:
   - Enhance data augmentation techniques
   - Experiment with different audio feature extraction methods
   - Implement efficient data loading for larger sample sizes

### Future Directions
1. **Multi-Label Evaluation**:
   - Implement specialized metrics for multi-label classification
   - Add class-wise metrics reporting

2. **Temporal Analysis**:
   - Incorporate temporal context in predictions
   - Analyze model performance across different times of day

3. **Deployment**:
   - Create inference pipeline for real-time audio processing
   - Convert models to TFLite for edge deployment

## Usage Instructions
1. Install required packages: `pip install -r requirements.txt`
2. Download the SONYC-UST dataset to `data/sonyc-ust/`
3. Run data validation: Execute cells in `pipeline.ipynb`
4. Train models: Use the training section in the notebook

## References
- [SONYC-UST Dataset](https://zenodo.org/record/3966543)
- [Soundata Documentation](https://soundata.readthedocs.io/)
- [Urban Sound Tagging Challenge](https://www.kaggle.com/c/dcase2019-task5/)