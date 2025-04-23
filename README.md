# Noise Bliss 
Denoising Algorithms for Acoustic Scene Classification in Deep Learning Models

### Audio is Multi-Class so changin los function is crucial

# TODOs:
- Schedule a meeting with Prof. Fuentes (Thu Apr 24 @10:30am EST)
    - models.py
    - Train a portion of a dataset $/rightarrow$ move forward

* Different branches for organization

# Classification 
    - Data Augmentation
    - Normalization
    - Windowing & Trimming
    - Spectral Analysis
    - Baseline Model
    - Train

# Temporal Analysis
    - Further divide test, validation, training out of temporal data scale
    - Call model for classification
        * Embedding

# Common Errors
- Using an autoencoder architecure where the input and output should mach is not efficient with a classification model (which outputs class probabilities)

# Different Kinds of Timeseries Tasks
A timeseries can be any data obtained via measurements at regular intervals.
- Understanding the dynamics of a system is crucial. By dynamics it is meant its periodic cycles, how it trends over time, its regular regime and its sudden spikes.
- For the SONY-UST dataset an event detection algorithm is crucial in order to identify the occurrence of a specific expected event within a continpus data audio stream for acoustic scene classification.
- Measuring periodicity in the data