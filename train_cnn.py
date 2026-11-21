import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    data_augmentation = models.Sequential([
        layers.RandomRotation(factor=0.05),
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        layers.RandomZoom(height_factor=0.08, width_factor=0.08),
    ], name="data_augmentation")

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="MNIST_CNN_Classifier")
    return model


def train_and_evaluate():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)

    model = build_cnn_model()
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs("models", exist_ok=True)
    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            "models/mnist_cnn_model.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=25,
        validation_split=0.1,
        callbacks=cb_list,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:\n", report)

    with open("models/training_metrics.json", "w") as f:
        json.dump(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "confusion_matrix": cm.tolist(),
                "history": {
                    k: [float(v) for v in vals] for k, vals in history.history.items()
                },
            },
            f,
            indent=2,
        )
    return history, report


if __name__ == "__main__":
    train_and_evaluate()
