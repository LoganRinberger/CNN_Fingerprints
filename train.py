"""Train the Siamese fingerprint matching model.

Run this file to teach the model that fingerprints from the same subject should
have nearby embeddings, while fingerprints from different subjects should have
far-apart embeddings.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FingerprintPairDataset, group_images_by_subject
from models.siamese_network import SiameseNetwork


DEFAULT_DATA_DIR = "data/SOCOFing/Real"
DEFAULT_CHECKPOINT = "checkpoints/siamese_fingerprint.pt"


class ContrastiveLoss(nn.Module):
    """Loss function for Siamese similarity learning.

    For matching pairs, it rewards small distances.
    For non-matching pairs, it rewards distances larger than the margin.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        # Matching pairs have label 1.0, so their distance should shrink.
        loss_match = label * torch.pow(distance, 2)

        # Non-matching pairs have label 0.0, so their distance should be at
        # least "margin". Once they are far enough apart, their loss is zero.
        loss_non_match = (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )

        return torch.mean(loss_match + loss_non_match)


def get_device():
    """Choose the best available device for PyTorch."""
    # Apple Silicon GPU support.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # NVIDIA GPU support.
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Universal fallback.
    return torch.device("cpu")


def split_subjects(real_dir, validation_fraction, seed):
    """Split subject IDs into train and validation groups.

    Splitting by subject is important. It means validation tests the model on
    people it did not see during training, which is more honest than mixing the
    same people across both sets.
    """
    image_paths = sorted(Path(real_dir).glob("*.BMP"))
    subject_to_images = group_images_by_subject(image_paths)
    subjects = sorted(subject_to_images)

    if len(subjects) < 3:
        raise ValueError("Need at least three subjects to create train and validation splits.")

    rng = random.Random(seed)
    rng.shuffle(subjects)

    # Use the first shuffled chunk for validation and the rest for training.
    validation_count = max(1, int(len(subjects) * validation_fraction))
    validation_subjects = subjects[:validation_count]
    train_subjects = subjects[validation_count:]

    return train_subjects, validation_subjects


def run_epoch(model, loader, criterion, optimizer, device):
    """Run one training pass over the DataLoader."""
    model.train()
    total_loss = 0.0

    for image1, image2, labels in loader:
        # Move tensors to the same device as the model before computing.
        image1 = image1.to(device)
        image2 = image2.to(device)
        labels = labels.to(device)

        # The Siamese model returns embeddings and their distance. Training only
        # needs the distance for contrastive loss.
        _, _, distance = model(image1, image2)
        loss = criterion(distance, labels)

        # Standard PyTorch training steps: clear old gradients, compute new
        # gradients, then update model weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold):
    """Measure validation loss and simple same/different accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for image1, image2, labels in loader:
        image1 = image1.to(device)
        image2 = image2.to(device)
        labels = labels.to(device)

        _, _, distance = model(image1, image2)
        loss = criterion(distance, labels)

        # Distances below the threshold are predicted as "same subject".
        predictions = (distance < threshold).float()

        total_loss += loss.item()
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader), correct / total


def save_checkpoint(path, model, optimizer, epoch, validation_loss, threshold, args):
    """Save everything needed to reuse or resume the trained model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "threshold": threshold,
            # Stored so future loading code knows what image size the model saw.
            "image_size": 128,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    """Define command-line settings for training experiments."""
    parser = argparse.ArgumentParser(description="Train a Siamese fingerprint matcher.")
    parser.add_argument("--real-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-pairs", type=int, default=4000)
    parser.add_argument("--validation-pairs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Create data, train the model, validate it, and save the best checkpoint."""
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    # Create a subject-level train/validation split before building datasets.
    train_subjects, validation_subjects = split_subjects(
        args.real_dir,
        args.validation_fraction,
        args.seed,
    )

    # These datasets generate random fingerprint pairs on demand.
    train_dataset = FingerprintPairDataset(
        real_dir=args.real_dir,
        num_pairs=args.train_pairs,
        subjects=train_subjects,
        seed=args.seed,
    )
    validation_dataset = FingerprintPairDataset(
        real_dir=args.real_dir,
        num_pairs=args.validation_pairs,
        subjects=validation_subjects,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    print(
        f"Training on {train_dataset.num_subjects} subjects "
        f"and validating on {validation_dataset.num_subjects} subjects."
    )

    # Model, loss, and optimizer are the three core parts of training.
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train once, then evaluate on held-out validation subjects.
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            args.threshold,
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.2%}"
        )

        # Keep the checkpoint with the best validation loss seen so far.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                epoch,
                validation_loss,
                args.threshold,
                args,
            )
            print(f"Saved checkpoint to {args.checkpoint}")

    print("Training complete.")


if __name__ == "__main__":
    main()
