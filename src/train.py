"""Train the plant disease classification CNN."""

import argparse
import json
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from src.data import create_data_generators
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_gen, valid_gen, _ = create_data_generators(
        args.data_dir, batch_size=args.batch_size
    )

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=len(train_gen.class_indices),
    )
    model.summary()

    best_model_path = model_dir / "best_plant_disease_cnn.keras"
    final_model_path = model_dir / "plant_disease_cnn.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(final_model_path)

    history_path = reports_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    class_map = {str(index): name for name, index in train_gen.class_indices.items()}
    with (reports_dir / "class_indices.json").open("w", encoding="utf-8") as file:
        json.dump(class_map, file, indent=2)

    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best checkpoint saved to: {best_model_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
