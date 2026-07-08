"""
Evaluation: per-class metrics, confusion matrix, ROC/AUC, training curves.

Why not just report accuracy: with 259 "yes" vs 22 "no" (~92% "yes"), a
model that predicts "yes" for every single image scores ~92% accuracy while
having zero ability to detect the "no" (no-tumor) class -- exactly the
class where a false "yes" (telling a healthy patient they have a tumor) or
a missed "no" pattern matters. Every function here reports per-class
precision/recall/F1 and the confusion matrix specifically so that failure
mode is visible instead of hidden behind a high headline number.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .data_utils import CLASS_NAMES


def get_predictions(model, dataset):
    """Runs the model over a tf.data eval dataset, returns y_true, y_prob, y_pred."""
    y_true, y_prob = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).ravel()
        y_true.extend(labels.numpy().tolist())
        y_prob.extend(probs.tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Returns overall accuracy plus per-class precision/recall/F1 (via
    sklearn's classification_report) and ROC-AUC. Kept as a plain dict so it
    can be dumped straight to JSON for results/ and read back by report.py.
    """
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": report["accuracy"],
        "auc": auc,
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
    }


def print_metrics(metrics: dict, model_name: str):
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}   ROC-AUC: {metrics['auc']:.3f}")
    print(f"{'class':<8}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for name in CLASS_NAMES:
        c = metrics["per_class"][name]
        print(f"{name:<8}{c['precision']:>10.3f}{c['recall']:>10.3f}{c['f1-score']:>10.3f}{c['support']:>10.0f}")


def save_metrics_json(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(y_true, y_pred, save_path: str, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    return cm


def plot_roc_curve(y_true, y_prob, save_path: str, model_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    return auc


def plot_training_curves(history, save_path: str, model_name: str):
    """Plots loss and accuracy for train vs val side by side from a Keras History object."""
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title(f"Loss — {model_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(h["accuracy"], label="train")
    axes[1].plot(h["val_accuracy"], label="val")
    axes[1].set_title(f"Accuracy — {model_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
