"""Dataset utilities for fingerprint similarity training.

This file turns SOCOFing fingerprint image files into pairs:
- label 1.0 means both images belong to the same subject
- label 0.0 means the images belong to different subjects

The Siamese network uses these pairs to learn when two fingerprints are close
or far apart in embedding space.
"""

import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# Every fingerprint is converted into the same model-ready tensor shape:
# 1 grayscale channel, 128 pixels tall, 128 pixels wide.
fingerprint_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def load_fingerprint_image(image_path):
    """Open one image file and convert it into a normalized PyTorch tensor."""
    with Image.open(image_path) as image:
        return fingerprint_transform(image)


def parse_socofing_filename(image_path):
    """Pull subject metadata out of a SOCOFing filename.

    Example filename:
    100__M_Left_index_finger.BMP

    Returns:
    - subject_id: "100"
    - gender: "M"
    - finger: "Left_index_finger"
    """
    image_path = Path(image_path)
    filename = image_path.stem

    # SOCOFing separates subject ID from the rest of the metadata with "__".
    parts = filename.split("__")
    if len(parts) != 2:
        raise ValueError(f"Unexpected SOCOFing filename format: {image_path.name}")

    subject_id = parts[0]
    remaining = parts[1]

    gender = remaining.split("_")[0]
    finger = "_".join(remaining.split("_")[1:])

    return subject_id, gender, finger


def get_subject_id(image_path):
    """Return only the subject/person ID from a fingerprint image path."""
    subject_id, _, _ = parse_socofing_filename(image_path)
    return subject_id


def group_images_by_subject(image_paths):
    """Build a lookup table like {"100": [image1, image2, ...], ...}.

    Grouping once up front makes pair creation much faster during training.
    """
    subject_to_images = defaultdict(list)

    for image_path in image_paths:
        subject_to_images[get_subject_id(image_path)].append(image_path)

    return dict(subject_to_images)


def create_pair(subject_to_images, subjects, rng=None):
    """Randomly create one labeled image pair.

    Half the time this tries to make a same-subject pair. The other half it
    makes a different-subject pair.
    """
    rng = rng or random

    # Pick the first image by choosing a subject, then one image for that subject.
    subject1 = rng.choice(subjects)
    image1_path = rng.choice(subject_to_images[subject1])

    same_person = rng.choice([True, False])

    if same_person:
        # For a match, the second image must be from the same subject but not
        # the exact same file.
        possible_matches = [path for path in subject_to_images[subject1] if path != image1_path]
        if not possible_matches:
            # If a subject only has one image, fall back to a non-match pair.
            same_person = False

    if same_person:
        image2_path = rng.choice(possible_matches)

        label = 1.0

    else:
        # For a non-match, choose a completely different subject.
        subject2 = rng.choice([subject for subject in subjects if subject != subject1])
        image2_path = rng.choice(subject_to_images[subject2])

        label = 0.0

    return image1_path, image2_path, label


class FingerprintPairDataset(Dataset):
    """PyTorch Dataset that returns fingerprint pairs instead of single images."""

    def __init__(self, real_dir, num_pairs=1000, subjects=None, seed=None):
        """Store dataset settings and prepare subject groups.

        real_dir: folder containing SOCOFing real fingerprint BMP files.
        num_pairs: how many random pairs this dataset should expose per epoch.
        subjects: optional subject ID list, useful for train/validation splits.
        seed: optional random seed for repeatable pair sampling.
        """
        self.real_dir = Path(real_dir)
        self.image_paths = sorted(self.real_dir.glob("*.BMP"))
        self.num_pairs = num_pairs
        self.rng = random.Random(seed) if seed is not None else random

        if not self.image_paths:
            raise FileNotFoundError(f"No BMP images found in {self.real_dir}")

        self.subject_to_images = group_images_by_subject(self.image_paths)

        # If a train/validation split gave us a subject list, keep only those
        # people in this dataset instance.
        if subjects is not None:
            subjects = [str(subject) for subject in subjects]
            self.subject_to_images = {
                subject: images
                for subject, images in self.subject_to_images.items()
                if subject in subjects
            }

        self.subjects = sorted(self.subject_to_images)

        if len(self.subjects) < 2:
            raise ValueError("At least two subjects are required to create non-matching pairs.")

    @property
    def num_subjects(self):
        """Number of people represented in this dataset."""
        return len(self.subjects)

    @property
    def num_images(self):
        """Number of image files represented in this dataset."""
        return sum(len(images) for images in self.subject_to_images.values())

    def __len__(self):
        """Tell PyTorch how many random pairs count as one epoch."""
        return self.num_pairs

    def __getitem__(self, index):
        """Create and load one random fingerprint pair."""
        image1_path, image2_path, label = create_pair(
            self.subject_to_images,
            self.subjects,
            self.rng,
        )

        image1 = load_fingerprint_image(str(image1_path))
        image2 = load_fingerprint_image(str(image2_path))

        label = torch.tensor(label, dtype=torch.float32)

        # Return format expected by the training loop:
        # first image tensor, second image tensor, same/different label.
        return image1, image2, label
