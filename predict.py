"""Compare two fingerprint images with a trained Siamese checkpoint."""

import argparse

import torch

from dataset import load_fingerprint_image
from models.siamese_network import SiameseNetwork
from train import DEFAULT_CHECKPOINT, get_device


def load_model(checkpoint_path, device):
    """Load trained model weights from disk and switch to evaluation mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SiameseNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_args():
    """Read image paths and optional settings from the command line."""
    parser = argparse.ArgumentParser(description="Compare two fingerprint images.")
    parser.add_argument("image1")
    parser.add_argument("image2")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    """Load images, compare them, and print the same/different prediction."""
    args = parse_args()
    device = get_device()

    model, checkpoint = load_model(args.checkpoint, device)

    # If the user does not pass a threshold, use the threshold saved during
    # training. Lower distances mean "more similar".
    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint.get("threshold", 0.5)

    # Add a batch dimension with unsqueeze(0), changing [1, 128, 128] into
    # [1, 1, 128, 128], which is what the model expects.
    image1 = load_fingerprint_image(args.image1).unsqueeze(0).to(device)
    image2 = load_fingerprint_image(args.image2).unsqueeze(0).to(device)

    # Compare the embeddings and convert the one-value tensor into a Python number.
    distance = model.compare(image1, image2).item()
    same_person = distance < threshold

    print("Using device:", device)
    print(f"Distance: {distance:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("Prediction:", "same subject" if same_person else "different subjects")


if __name__ == "__main__":
    main()
