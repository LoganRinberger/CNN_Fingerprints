"""Quick smoke test for the dataset, CNN encoder, and Siamese network.

This is not a full accuracy test. It simply confirms that the main pieces can
run together and produce tensors with the expected shapes.
"""

import torch

from dataset import FingerprintPairDataset
from models.fingerprint_cnn import FingerprintCNN
from models.siamese_network import SiameseNetwork
from train import ContrastiveLoss, get_device


def main():
    """Load one fingerprint pair and pass it through the model pipeline."""
    device = get_device()
    print("Using device:", device)

    # Build a tiny dataset that can generate random same/different pairs.
    dataset = FingerprintPairDataset(
        real_dir="data/SOCOFing/Real",
        num_pairs=8,
        seed=7,
    )
    image1, image2, label = dataset[0]

    # Test the single-image encoder first.
    encoder = FingerprintCNN().to(device)
    embedding = encoder(image1.unsqueeze(0).to(device))
    print("Encoder output shape:", embedding.shape)

    # Test the full Siamese comparison model next.
    model = SiameseNetwork().to(device)

    # unsqueeze(0) adds the batch dimension expected by PyTorch models.
    image1_batch = image1.unsqueeze(0).to(device)
    image2_batch = image2.unsqueeze(0).to(device)
    label_batch = label.unsqueeze(0).to(device)

    _, _, distance = model(image1_batch, image2_batch)

    # Loss should produce one scalar value for this single pair.
    loss = ContrastiveLoss()(distance, label_batch)

    print("Siamese distance shape:", distance.shape)
    print("Sample label:", label.item())
    print("Sample distance:", distance.item())
    print("Sample loss:", loss.item())


if __name__ == "__main__":
    main()
