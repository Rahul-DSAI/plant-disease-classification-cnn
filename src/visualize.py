"""Plot training and validation accuracy/loss from a Keras history JSON file."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        default="reports/training_history.json",
        help="Path to training_history.json",
    )
    parser.add_argument(
        "--output",
        default="reports/training_curves.png",
        help="Output image path",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    output_path = Path(args.output)
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    required = ("accuracy", "val_accuracy", "loss", "val_loss")
    missing = [key for key in required if key not in history]
    if missing:
        raise ValueError(f"History is missing required keys: {', '.join(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["accuracy"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Model accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["loss"], label="Training loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation loss")
    axes[1].set_title("Model loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Training curves saved to: {output_path}")


if __name__ == "__main__":
    main()
