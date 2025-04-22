# noise-bliss
Denoising Algorithms for Acoustic Scene Classification in Deep Learning Models

### Audio is Multi-Class so changin los function is crucial

TODO:
* Schedule a meeting with Prof. Fuentes
    * models.py
    * Train a portion of a dataset $/rightarrow$ move forward

* Different branches for organization

* Classification 
    * Data Augmentation
    * Normalization
    * Windowing & Trimming
    * Spectral Analysis
    * Baseline Model
    * Train

* Temporal Analysis
    * Further divide test, validation, training out of temporal data scale
    * Call model for classification
        * Embedding

# Common Errors
- Using an autoencoder architecure where the input and output should mach is not efficient with a classification model (which outputs class probabilities)