"""CNN model definition."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """Build the baseline three-block CNN from the original project."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ],
        name="plant_disease_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
