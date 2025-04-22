import tensorflow as tf
from tensorflow import keras 
from keras import layers

def autoencoder_model(input_shape):
    """
    Creates an autoencoder model for audio denoising
    Encoder + Decoder = Model
    """
    model = keras.Sequential([
        
        # ENCODER
        # first convolutional block
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # DECODER
        # third convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.UpSampling2D((2, 2)),

        # fourth convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.UpSampling2D((2, 2)),

        # output
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

def compile_model(model):
    """
    Compiles the model with appropriate loss and metrics fot multi-label
    classification.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        # using mse for autoencoder
        loss='mse',
        # mean absolute error
        metrics=['mae']
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