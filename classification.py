# ============================================ #
#               IMPORTS
# ============================================ #
import os
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    optimizers,
    metrics,
    regularizers,
    Input,
    Model
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Concatenate,
    Conv1D,
    GlobalAveragePooling1D
)

from utils import augment_spectrogram  # Custom utility for data augmentation

# ============================================ #
#            MODEL DEFINITIONS
# ============================================ #

def cnn_model(input_shape, num_classes):
    """
    Build a CNN model for multi-label audio classification (spectrogram input).
    
    Args:
        input_shape (tuple): (height, width, channels) of input spectrograms.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: CNN model instance.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Fully connected head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


def tcn_block(input_layer, nb_filters, kernel_size, dilation_rate):
    """
    Single block of a Temporal Convolutional Network (TCN).
    
    Args:
        input_layer (tensor): Input tensor.
        nb_filters (int): Number of convolution filters.
        kernel_size (int): Size of convolution kernel.
        dilation_rate (int): Dilation rate.

    Returns:
        tensor: Output tensor after Conv1D, BatchNorm, and Dropout.
    """
    x = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu'
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    return x


def tcn_model(input_shape, num_classes, nb_filters=64, kernel_size=3, nb_stacks=1):
    """
    Build a TCN model for sequential audio feature learning.

    Args:
        input_shape (tuple): (time_steps, features) of input.
        num_classes (int): Number of output classes.
        nb_filters (int): Filters per convolution.
        kernel_size (int): Kernel size.
        nb_stacks (int): Number of TCN block stacks.

    Returns:
        keras.Model: TCN model instance.
    """
    input_layer = layers.Input(shape=input_shape)
    x = input_layer

    # Stack multiple TCN blocks with increasing dilation
    for _ in range(nb_stacks):
        for d in [1, 2, 4, 8]:
            x = tcn_block(x, nb_filters, kernel_size, d)

    # Fully connected head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='sigmoid')(x)

    return keras.Model(input_layer, output_layer)

# Build multimodal Conv1D + metadata model
def build_multimodal_model(audio_shape, temporal_shape, num_classes):
    audio_input = Input(shape=audio_shape, name="audio_input")
    x_audio = Conv1D(64, kernel_size=3, padding="causal", activation="relu")(audio_input)
    x_audio = GlobalAveragePooling1D()(x_audio)

    temporal_input = Input(shape=(temporal_shape,), name="temporal_input")
    x_temp = Dense(32, activation="relu")(temporal_input)

    x = Concatenate()([x_audio, x_temp])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="sigmoid")(x)

    return Model(inputs=[audio_input, temporal_input], outputs=output)



# ============================================ #
#         COMPILATION AND TRAINING
# ============================================ #

def compile_model(model, learning_rate=0.001):
    """
    Compile the model for multi-label classification with binary crossentropy.

    Args:
        model (keras.Model): Keras model.
        learning_rate (float): Learning rate.

    Returns:
        keras.Model: Compiled model.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
            metrics.AUC(name='auc', multi_label=True)
        ]
    )
    return model


def train_model(model, train_data, val_data, epochs=15, batch_size=None, steps_per_epoch=None, validation_steps=None):
    """
    Train model with early stopping, LR reduction, and checkpointing.

    Args:
        model (keras.Model): Model to train.
        train_data (tf.data.Dataset): Training dataset.
        val_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs.
        batch_size (int, optional): Batch size.
        steps_per_epoch (int, optional): Number of steps per epoch.
        validation_steps (int, optional): Number of steps for validation.

    Returns:
        keras.callbacks.History: Training history object.
    """
    model_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(filepath='best_model_baseline.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        batch_size=batch_size,
        callbacks=model_callbacks,
        verbose=1
    )

    return history

def class_weighted_augmentation(spec, label, class_weights):
    """
    Conditionally augment a spectrogram based on label imbalance score.

    Parameters
    ----------
    spec : tf.Tensor
        Input mel spectrogram (H x W x 1).
    label : tf.Tensor
        Multi-hot encoded label vector.
    class_weights : tf.Tensor
        Class weighting vector for imbalance-aware augmentation.

    Returns
    -------
    tuple (tf.Tensor, tf.Tensor)
        Possibly augmented spectrogram and its label.
    """
    imbalance_score = tf.reduce_sum(label * class_weights)
    prob = tf.math.sigmoid(imbalance_score - 1.0)
    random_value = tf.random.uniform([], 0, 1)

    # Both branches must be lambdas with no arguments
    return tf.cond(
        random_value < prob,
        lambda: (augment_spectrogram(spec), label),
        lambda: (spec, label)
    )


def build_cnn_model(input_shape=(128, 128, 1), num_classes=8):
    """
    Build CNN model for multi-label spectrogram classification.

    Args:
        input_shape (tuple): Input spectrogram dimensions.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: CNN model instance.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="sigmoid")
    ])
    return model


def compile_augmented_model(model, learning_rate=0.001):
    """
    Compile model for training with binary crossentropy and AUC metrics.

    Args:
        model (keras.Model): Keras model.
        learning_rate (float): Learning rate.

    Returns:
        keras.Model: Compiled model.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            metrics.AUC(name="auc", multi_label=True)
        ]
    )
    return model


def create_augmented_dataset(dataset, class_weights):
    """
    Create augmented dataset by applying class-weighted augmentation.

    Args:
        dataset (tf.data.Dataset): Original dataset (spectrogram, label).
        class_weights (list or np.array): Class imbalance weights.

    Returns:
        tf.data.Dataset: Augmented dataset.
    """
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    return dataset.map(
        lambda spec, label: class_weighted_augmentation(spec, label, class_weights_tensor),
        num_parallel_calls=tf.data.AUTOTUNE
    )

# ============================================ #
#       CUSTOM LOSS FUNCTION
# ============================================ #

def get_weighted_binary_crossentropy(class_weights):
    """
    Create a custom weighted binary cross-entropy loss for imbalanced multi-label classification.

    Args:
        class_weights (list or np.array): Per-class weights.

    Returns:
        function: Custom loss function.
    """
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weighted_bce = bce * class_weights_tensor
        return tf.reduce_mean(weighted_bce, axis=-1)

    return loss_fn