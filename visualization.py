"""Generate diagrams for fingerprint similarity scores.

This script loads a trained Siamese checkpoint, creates random fingerprint
pairs, measures their distances, and saves Matplotlib diagrams into diagrams/.
"""

import argparse
import os
from pathlib import Path

# Matplotlib wants a writable cache directory. Setting this before importing
# pyplot avoids warnings on machines where the default cache folder is locked.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

# Use a non-window backend so diagrams are reliably written to files. Interactive
# windows can be unreliable from IDE terminals and sandboxed shells.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FingerprintPairDataset
from predict import load_model
from train import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, get_device


def collect_similarity_scores(model, loader, device):
    """Run fingerprint pairs through the model and collect labels/distances."""
    labels = []
    distances = []
    examples = []

    model.eval()
    with torch.no_grad():
        for image1, image2, batch_labels in loader:
            image1 = image1.to(device)
            image2 = image2.to(device)

            batch_distances = model.compare(image1, image2).cpu()

            labels.extend(batch_labels.tolist())
            distances.extend(batch_distances.tolist())

            # Keep a few raw image tensors for the example-pair grid.
            if len(examples) < 12:
                for index in range(image1.size(0)):
                    examples.append(
                        {
                            "image1": image1[index].cpu(),
                            "image2": image2[index].cpu(),
                            "label": float(batch_labels[index].item()),
                            "distance": float(batch_distances[index].item()),
                        }
                    )
                    if len(examples) >= 12:
                        break

    return np.array(labels), np.array(distances), examples


def plot_distance_histogram(labels, distances, threshold, output_path):
    """Save a histogram showing same-subject vs different-subject distances."""
    same_distances = distances[labels == 1.0]
    different_distances = distances[labels == 0.0]

    plt.figure(figsize=(10, 6))
    plt.hist(same_distances, bins=25, alpha=0.7, label="Same subject", color="#2a9d8f")
    plt.hist(
        different_distances,
        bins=25,
        alpha=0.7,
        label="Different subjects",
        color="#e76f51",
    )
    plt.axvline(threshold, color="#264653", linestyle="--", label=f"Threshold {threshold:.2f}")
    plt.xlabel("Embedding distance")
    plt.ylabel("Number of pairs")
    plt.title("Fingerprint Similarity Distance Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_distance_scatter(labels, distances, threshold, output_path):
    """Save a scatter plot of each pair's distance against the threshold."""
    pair_numbers = np.arange(len(distances))
    colors = np.where(labels == 1.0, "#2a9d8f", "#e76f51")

    plt.figure(figsize=(10, 6))
    plt.scatter(pair_numbers, distances, c=colors, alpha=0.75, edgecolors="none")
    plt.axhline(threshold, color="#264653", linestyle="--", label=f"Threshold {threshold:.2f}")
    plt.xlabel("Generated pair number")
    plt.ylabel("Embedding distance")
    plt.title("Similarity Scores by Fingerprint Pair")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", label="Same subject", markerfacecolor="#2a9d8f", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Different subjects", markerfacecolor="#e76f51", markersize=8),
        plt.Line2D([0], [0], color="#264653", linestyle="--", label=f"Threshold {threshold:.2f}"),
    ])
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_summary(labels, distances, threshold, output_path):
    """Save a small summary chart with average distances and accuracy."""
    same_distances = distances[labels == 1.0]
    different_distances = distances[labels == 0.0]
    predictions = (distances < threshold).astype(float)
    accuracy = (predictions == labels).mean()

    names = ["Same subject avg", "Different avg", "Accuracy"]
    values = [
        same_distances.mean() if len(same_distances) else 0,
        different_distances.mean() if len(different_distances) else 0,
        accuracy,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#2a9d8f", "#e76f51", "#457b9d"])
    plt.title("Similarity Score Summary")
    plt.ylabel("Distance / Accuracy")
    plt.ylim(0, max(1.0, max(values) * 1.2))

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_example_pairs(examples, threshold, output_path):
    """Save a visual grid of fingerprint pairs and their model scores."""
    rows = len(examples)
    fig, axes = plt.subplots(rows, 2, figsize=(6, max(3, rows * 2)))

    if rows == 1:
        axes = np.array([axes])

    for row, example in enumerate(examples):
        label_text = "same" if example["label"] == 1.0 else "different"
        prediction_text = "same" if example["distance"] < threshold else "different"

        for column, key in enumerate(["image1", "image2"]):
            axes[row, column].imshow(example[key].squeeze(0), cmap="gray")
            axes[row, column].axis("off")

        axes[row, 0].set_title(f"True: {label_text}", fontsize=9)
        axes[row, 1].set_title(
            f"Pred: {prediction_text} | d={example['distance']:.3f}",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args():
    """Read visualization settings from the command line."""
    parser = argparse.ArgumentParser(description="Visualize fingerprint similarity scores.")
    parser.add_argument("--real-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default="diagrams")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="Open each diagram window after saving.")
    return parser.parse_args()


def main():
    """Generate and save all visualization diagrams."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, checkpoint = load_model(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)

    dataset = FingerprintPairDataset(
        real_dir=args.real_dir,
        num_pairs=args.pairs,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    labels, distances, examples = collect_similarity_scores(model, loader, device)

    outputs = [
        output_dir / "distance_histogram.png",
        output_dir / "distance_scatter.png",
        output_dir / "similarity_summary.png",
        output_dir / "example_pairs.png",
    ]

    plot_distance_histogram(labels, distances, threshold, outputs[0])
    plot_distance_scatter(labels, distances, threshold, outputs[1])
    plot_summary(labels, distances, threshold, outputs[2])
    plot_example_pairs(examples, threshold, outputs[3])

    print("Saved diagrams:")
    for output in outputs:
        print(f"- {output}")

    if args.show:
        for output in outputs:
            image = plt.imread(output)
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.axis("off")
            plt.title(output.name)
            plt.show()


if __name__ == "__main__":
    main()
