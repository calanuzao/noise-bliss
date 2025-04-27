"""
models.py

Defines the following models and training utilities:
- CNN model for spectrogram input
- CNN + Metadata Fusion model
- Training function with callbacks
"""

# ============================
# Imports
# ============================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, metrics, Input, Model
import numpy as np

# ============================
# CNN Model for Audio Only
# ============================

def cnn_model2(input_shape=(128, 128, 1), num_classes=8):
    """
    CNN model for urban sound classification.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input mel spectrograms (height, width, channels)
    num_classes : int
        Number of sound classes to predict
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer - multi-label classification
        layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC()]
    )

    return model

def cnn_model(input_shape=(128, 128, 1), num_classes=8):
    """
    CNN model for urban sound classification.

    Parameters
    ----------
    input_shape : tuple
        Shape of input mel spectrograms (height, width, channels)
    num_classes : int
        Number of sound classes to predict

    Returns
    -------
    keras.Model
        Compiled CNN model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output layer with sigmoid activation for multi-label classification
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')]
    )

    return model

# ============================
# Training Utility Function
# ============================

def train_model(model, train_data, validation_data, epochs, steps_per_epoch=None, validation_steps=None):
    """
    Train the model with callbacks for early stopping and model checkpointing.

    Parameters
    ----------
    model : keras.Model
        The compiled model to train
    train_data : tf.data.Dataset
        Training dataset
    validation_data : tf.data.Dataset
        Validation dataset
    epochs : int
        Number of epochs to train
    steps_per_epoch : int, optional
        Number of steps per epoch
    validation_steps : int, optional
        Number of steps per validation epoch

    Returns
    -------
    history : keras.callbacks.History
        Training history object
    """
    model_callbacks = [
        # Stop training if validation loss doesn't improve
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model based on validation loss
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        # Reduce learning rate if validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=model_callbacks,
        verbose=1
    )

    return history

# ============================
# CNN + Metadata Fusion Model
# ============================

def fusion_cnn_metadata_model(audio_input_shape=(128, 128, 1), metadata_input_shape=(4,), num_classes=8):
    """
    Builds a CNN + Metadata Fusion Model for multi-label classification.

    Parameters
    ----------
    audio_input_shape : tuple
        Shape of the input spectrogram (default (128, 128, 1)).
    metadata_input_shape : tuple
        Shape of the input metadata vector.
    num_classes : int
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        Compiled fusion model
    """
    # Audio Branch: CNN to extract features from spectrogram
    audio_input = Input(shape=audio_input_shape, name="audio_input")
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(audio_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Metadata Branch: MLP to process structured metadata
    metadata_input = Input(shape=metadata_input_shape, name="metadata_input")
    m = layers.Dense(32, activation='relu')(metadata_input)
    m = layers.Dense(64, activation='relu')(m)

    # Fusion of audio features and metadata features
    combined = layers.Concatenate()([x, m])

    # Fully connected layers after fusion
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(num_classes, activation='sigmoid')(z)

    model = Model(inputs=[audio_input, metadata_input], outputs=output)

    return model