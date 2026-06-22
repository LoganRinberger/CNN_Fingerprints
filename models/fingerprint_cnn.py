"""CNN encoder that converts one fingerprint image into an embedding vector."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerprintCNN(nn.Module):
    """Small convolutional network for fingerprint feature extraction.

    The network does not classify a person directly. Instead, it converts a
    fingerprint image into a 64-number embedding. The Siamese network compares
    two of these embeddings to decide whether two fingerprints look related.
    """

    def __init__(self):
        super(FingerprintCNN, self).__init__()

        # Convolution blocks learn visual fingerprint patterns such as ridges,
        # curves, endings, and crossings. MaxPool halves the image size after
        # each block: 128 -> 64 -> 32 -> 16.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # After the convolution blocks, the image has become 64 feature maps of
        # size 16x16. This section flattens that into a 64-value embedding.
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        """Run a batch of fingerprint tensors through the encoder."""
        x = self.features(x)
        x = self.embedding(x)

        # Unit-length embeddings make distance comparisons more consistent.
        return F.normalize(x, p=2, dim=1)
