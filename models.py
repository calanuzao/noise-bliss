import tensorflow as tf
from tensorflow import keras 
from keras import layers

def classification_model(input_shape, num_classes):
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

def compile_model(model):
    """
    Compiles the model with appropriate loss and metrics fot multi-label
    classification.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]  
    )
    return model

def train_model(model, train_data, validation_data, epochs=50):
    """
    Trains the model with early stopping
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return history