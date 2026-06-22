"""Siamese network for comparing two fingerprint images."""

import torch.nn as nn
import torch.nn.functional as F

from models.fingerprint_cnn import FingerprintCNN


class SiameseNetwork(nn.Module):
    """Use one shared encoder to compare two fingerprints.

    "Siamese" means both input images pass through the exact same CNN. Sharing
    weights forces both images to be measured by the same feature extractor.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # One encoder instance is reused for both images.
        self.encoder = FingerprintCNN()

    def forward(self, image1, image2):
        """Return both embeddings and the distance between them."""
        embedding1 = self.encoder(image1)
        embedding2 = self.encoder(image2)

        # Smaller distance means the fingerprints look more similar.
        distance = F.pairwise_distance(embedding1, embedding2)

        return embedding1, embedding2, distance

    def compare(self, image1, image2):
        """Convenience helper for prediction code that only needs distance."""
        _, _, distance = self.forward(image1, image2)
        return distance
