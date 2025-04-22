# classification model

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