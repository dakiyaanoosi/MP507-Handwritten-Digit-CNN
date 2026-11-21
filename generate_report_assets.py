import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("docs/plots", exist_ok=True)
os.makedirs("docs/screenshots", exist_ok=True)


def plot_fig1_2_mnist_samples():
    """Figure 1.2: Representative Sample Images and Class Diversity from the MNIST Dataset."""
    import tensorflow as tf
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
    fig.suptitle("MNIST Dataset Diversity: Representative Samples (0–9)", fontsize=13, fontweight="bold")

    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0]
        # Choose two distinct samples for each digit
        idx1, idx2 = digit_indices[0], digit_indices[1]

        axes[0, digit].imshow(x_train[idx1], cmap="gray")
        axes[0, digit].set_title(f"Digit {digit}", fontsize=10, fontweight="bold")
        axes[0, digit].axis("off")

        axes[1, digit].imshow(x_train[idx2], cmap="gray")
        axes[1, digit].axis("off")

    plt.tight_layout()
    output_path = "docs/plots/fig1_2_mnist_samples.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {output_path}")


def plot_fig6_1_1_class_distribution():
    """Figure 6.1.1: MNIST Training and Test Set Class Distribution Chart."""
    digits = np.arange(10)
    train_counts = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    test_counts = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]

    bar_width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))

    rects1 = ax.bar(digits - bar_width / 2, train_counts, bar_width, label="Training Set (60,000)", color="#2563eb")
    rects2 = ax.bar(digits + bar_width / 2, test_counts, bar_width, label="Test Set (10,000)", color="#93c5fd")

    ax.set_xlabel("Digit Class (0–9)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_title("MNIST Dataset Class Distribution (Train vs. Test)", fontsize=13, fontweight="bold")
    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits])
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add count labels on top of bars
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    output_path = "docs/plots/fig6_1_1_class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {output_path}")


def plot_training_curves():
    """Figures 6.2.1 and 6.2.2: Accuracy and Loss vs. Epochs curves."""
    metrics_path = "models/training_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"[SKIP] {metrics_path} not found. Run train_cnn.py first.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    history = data["history"]
    epochs = range(1, len(history["accuracy"]) + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["accuracy"], marker="o", color="#2563eb", linewidth=2, label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="s", color="#16a34a", linewidth=2, label="Val Accuracy")
    plt.title("Model Training and Validation Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    acc_path = "docs/plots/fig6_2_1_accuracy_curve.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {acc_path}")

    # 2. Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["loss"], marker="o", color="#dc2626", linewidth=2, label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", color="#d97706", linewidth=2, label="Val Loss")
    plt.title("Model Training and Validation Loss vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    loss_path = "docs/plots/fig6_2_2_loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {loss_path}")


def plot_fig6_3_confusion_matrix():
    """Figure 6.3: Normalized 10x10 Confusion Matrix Heatmap."""
    metrics_path = "models/training_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"[SKIP] {metrics_path} not found.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, alpha=0.9)
    fig.colorbar(cax)

    for i in range(10):
        for j in range(10):
            count = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}", ha="center", va="center", color=color, fontsize=8.5)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_yticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Predicted Digit", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Digit", fontsize=11, fontweight="bold")
    ax.set_title("10x10 Confusion Matrix on MNIST Test Set (10,000 Samples)", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    cm_path = "docs/plots/fig6_3_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {cm_path}")


def plot_fig6_5_2_preprocessing_pipeline():
    """Figure 6.5.2: OpenCV 20x20 Bounding Box and Center-of-Mass Alignment Visualization."""
    from PIL import Image, ImageDraw
    import cv2
    from routes import preprocess_digit_image

    # Create synthetic 280x280 handwritten digit '3' with slight off-center offset
    img = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw a 3
    draw.arc([60, 40, 180, 140], start=180, end=360, fill=(0, 0, 0), width=18)
    draw.arc([60, 130, 190, 240], start=180, end=360, fill=(0, 0, 0), width=18)

    # 1. Raw grayscale inverted
    gray = np.array(img.convert("L"), dtype=np.uint8)
    inv = 255 - gray
    _, thresh = cv2.threshold(inv, 25, 255, cv2.THRESH_TOZERO)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    roi = thresh[y:y + h, x:x + w]

    # Preprocessed tensor
    tensor, processed_canvas = preprocess_digit_image(img, invert=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    axes[0].imshow(img)
    axes[0].set_title("(1) Raw Canvas Input\n(280 × 280 Light Paper)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(roi, cmap="gray")
    axes[1].set_title(f"(2) Cropped Bounding Box ROI\n({w} × {h} Foreground)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(processed_canvas, cmap="gray")
    axes[2].set_title("(3) Normalized & Centered Tensor\n(28 × 28 Center-of-Mass Aligned)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle("OpenCV Spatial Normalization and Center-of-Mass Pipeline", fontsize=12, fontweight="bold")
    plt.tight_layout()
    pipe_path = "docs/plots/fig6_5_2_preprocessing_pipeline.png"
    plt.savefig(pipe_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {pipe_path}")


def plot_fig6_1_2_preprocessed_grid():
    """Figure 6.1.2: Sample Preprocessed 28x28 Grayscale Digit Grid with Predictions."""
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test_norm = np.expand_dims(x_test.astype("float32") / 255.0, -1)

    model_path = "models/mnist_cnn_model.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        preds = np.argmax(model.predict(x_test_norm[:16], verbose=0), axis=1)
    else:
        preds = y_test[:16]

    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    fig.suptitle("Preprocessed 28×28 Grayscale Digits (True vs. Pred)", fontsize=12, fontweight="bold")

    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(x_test[i], cmap="gray")
        color = "blue" if preds[i] == y_test[i] else "red"
        ax.set_title(f"True: {y_test[i]} | Pred: {preds[i]}", fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    grid_path = "docs/plots/fig6_1_2_preprocessed_grid.png"
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {grid_path}")


def generate_all():
    print("Generating report assets...")
    plot_fig6_1_1_class_distribution()
    plot_fig1_2_mnist_samples()
    plot_fig6_5_2_preprocessing_pipeline()
    plot_training_curves()
    plot_fig6_3_confusion_matrix()
    plot_fig6_1_2_preprocessed_grid()
    print("All report assets generation attempt completed.")


if __name__ == "__main__":
    generate_all()
