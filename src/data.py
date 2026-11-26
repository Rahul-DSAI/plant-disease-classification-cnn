"""Dataset loading utilities for directory-based image classification."""

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (224, 224)


def create_data_generators(data_dir, batch_size=32, image_size=IMAGE_SIZE):
    """Create train, validation, and test generators from class subdirectories."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    for directory in (train_dir, valid_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required dataset directory not found: {directory}"
            )

    # Keep preprocessing consistent with the original implementation:
    # pixel values are rescaled to [0, 1]; no augmentation is applied.
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    common = {
        "target_size": image_size,
        "batch_size": batch_size,
        "class_mode": "categorical",
    }

    train_generator = datagen.flow_from_directory(
        str(train_dir), shuffle=True, **common
    )
    valid_generator = datagen.flow_from_directory(
        str(valid_dir), shuffle=False, **common
    )
    test_generator = datagen.flow_from_directory(
        str(test_dir), shuffle=False, **common
    )

    if train_generator.class_indices != valid_generator.class_indices:
        raise ValueError("Training and validation class folders do not match.")
    if train_generator.class_indices != test_generator.class_indices:
        raise ValueError("Training and test class folders do not match.")

    return train_generator, valid_generator, test_generator
