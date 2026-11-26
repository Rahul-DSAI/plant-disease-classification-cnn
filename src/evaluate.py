"""Evaluate a saved model on the held-out test directory."""

import argparse

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/plant_disease_cnn.keras")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model)

    generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        args.test_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    loss, accuracy = model.evaluate(generator, verbose=1)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    probabilities = model.predict(generator, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = generator.classes
    class_names = [
        name for name, index in sorted(
            generator.class_indices.items(), key=lambda item: item[1]
        )
    ]

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        print("\nClassification report:")
        print(
            classification_report(
                true_labels,
                predictions,
                labels=list(range(len(class_names))),
                target_names=class_names,
                zero_division=0,
            )
        )
        print("Confusion matrix:")
        print(confusion_matrix(true_labels, predictions))
    except ImportError:
        print("Install scikit-learn to display the classification report.")


if __name__ == "__main__":
    main()
