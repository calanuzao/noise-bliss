import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def cnn_model(input_shape, num_classes):
    """
    Creates a CNN model for multi-label audio classification.
        Three convolutional blocks
    
    Args:
        input_shape: Tuple of (height, width, channels) for input spectrograms
        num_classes: Number of sound classes to predict
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # convolutional blocks. filters with the following order -> 32 -> 64 -> 128
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # https://keras.io/api/layers/normalization_layers/batch_normalization/
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # https://keras.io/api/layers/normalization_layers/batch_normalization/
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        # https://keras.io/api/layers/normalization_layers/batch_normalization/
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # output layer with multi-label classification
        layers.Dense(num_classes, activation='sigmoid')
    ])
    return model 

def tcn_block(input_layer, nb_filters, kernel_size, dilation_rate):
    """
    Single TCN block with dilated convolutions
    """
    padding = (kernel_size - 1) * dilation_rate
    pad_layer = layers.ZeroPadding1D(padding)
    conv_layer = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal'
    )
    
    x = pad_layer(input_layer)
    x = conv_layer(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    return x

def tcn_model(input_shape, num_classes, nb_filters=64, kernel_size=3, nb_stacks=1):
    """
    Creates a Temporal Convolutional Network (TCN) model.
    
    Args:
        input_shape: Tuple of (timesteps, features) for input spectrograms
        num_classes: Number of sound classes to predict
        nb_filters: Number of filters in conv layers
        kernel_size: Size of the convolutional kernel
        nb_stacks: Number of TCN stacks
    """
    input_layer = layers.Input(shape=input_shape)
    x = input_layer

    for stack in range(nb_stacks):
        for d in [1, 2, 4, 8]:  # Dilations
            x = tcn_block(x, nb_filters, kernel_size, dilation_rate=d)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = keras.Model(input_layer, output_layer)
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile model with standard settings for multi-label classification
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def train_model(model, train_data, val_data, epochs=50, batch_size=32):
    """
    Train model with early stopping and learning rate reduction
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    ]
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history