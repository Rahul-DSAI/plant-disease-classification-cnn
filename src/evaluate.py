"""Evaluate a saved model on a held-out test directory and save diagnostics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best_plant_disease_cnn.keras")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reports-dir", default="reports")
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
        name
        for name, index in sorted(
            generator.class_indices.items(), key=lambda item: item[1]
        )
    ]

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        labels = list(range(len(class_names)))
        print("\nClassification report:")
        print(
            classification_report(
                true_labels,
                predictions,
                labels=labels,
                target_names=class_names,
                zero_division=0,
            )
        )
        matrix = confusion_matrix(true_labels, predictions, labels=labels)

        reports_dir = Path(args.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        figure_width = max(8, min(18, len(class_names) * 0.65))
        figure, axis = plt.subplots(
            figsize=(figure_width, figure_width * 0.8)
        )
        image = axis.imshow(matrix, interpolation="nearest")
        axis.set_title("Test set confusion matrix")
        figure.colorbar(image, ax=axis)
        ticks = np.arange(len(class_names))
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.set_xticklabels(class_names, rotation=90, fontsize=7)
        axis.set_yticklabels(class_names, fontsize=7)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        figure.tight_layout()
        output_path = reports_dir / "confusion_matrix.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"Confusion matrix saved to: {output_path}")
    except ImportError:
        print("Install scikit-learn to display the classification report.")


if __name__ == "__main__":
    main()
