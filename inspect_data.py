"""Inspect one batch of generated fingerprint pairs.

Use this file when you want to quickly confirm that the dataset is finding
images, generating labels, and returning tensors with the expected shapes.
"""

from torch.utils.data import DataLoader

from dataset import FingerprintPairDataset

# Build a small pair dataset from the real SOCOFing images.
dataset = FingerprintPairDataset(
    real_dir="data/SOCOFing/Real",
    num_pairs=100,
    seed=42,
)

# DataLoader batches multiple random pairs together.
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# Pull only the first batch so this script stays quick.
image1, image2, labels = next(iter(loader))

# These printouts verify both the dataset size and the batch tensor shapes.
print("Subjects:", dataset.num_subjects)
print("Images:", dataset.num_images)
print("Image 1 batch shape:", image1.shape)
print("Image 2 batch shape:", image2.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)
