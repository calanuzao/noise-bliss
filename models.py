import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models, callbacks, optimizers, metrics 
import numpy as np

def cnn_model(input_shape=(128, 128, 1), num_classes=8):
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

def train_model(model, train_data, validation_data, epochs=50):
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
        
    Returns
    -------
    history : keras.callbacks.History
        Training history
    """
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=model_callbacks,
        verbose=1
    )
    
    return history