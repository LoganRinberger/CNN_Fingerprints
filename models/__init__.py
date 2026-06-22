"""Convenient imports for the model package."""

# These exports let other files import directly from "models" if desired:
# from models import FingerprintCNN, SiameseNetwork
from .fingerprint_cnn import FingerprintCNN
from .siamese_network import SiameseNetwork

__all__ = ["FingerprintCNN", "SiameseNetwork"]
