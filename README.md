# CNN Fingerprints

CNN Fingerprints is a PyTorch project for learning fingerprint similarity with a Siamese neural network. Instead of classifying a fingerprint as a specific person, the model compares two fingerprint images and predicts whether they appear to belong to the same subject.

The current dataset workflow is built around the SOCOFing fingerprint dataset. Images are loaded as grayscale `128x128` tensors, paired into same-subject or different-subject examples, and passed through a shared CNN encoder. The distance between the two learned embeddings becomes the similarity score.

## Current Features

- Loads SOCOFing `.BMP` fingerprint images.
- Parses subject IDs from SOCOFing filenames.
- Creates random same-subject and different-subject training pairs.
- Trains a Siamese CNN with contrastive loss.
- Splits training and validation by subject, not by random image, for a more honest validation check.
- Saves model checkpoints.
- Compares two fingerprint images with a trained checkpoint.
- Generates Matplotlib visualizations for similarity scores, threshold behavior, summary metrics, and example fingerprint pairs.

## Project Structure

```text
.
├── dataset.py
├── inspect_data.py
├── predict.py
├── requirements.txt
├── test.py
├── train.py
├── visualization.py
└── models/
    ├── __init__.py
    ├── fingerprint_cnn.py
    └── siamese_network.py
```

## File Overview

### `dataset.py`

Contains the data loading and pair generation logic. It converts fingerprint images into PyTorch tensors, parses SOCOFing filenames, groups images by subject, and creates labeled pairs:

- `1.0` means same subject.
- `0.0` means different subjects.

The main class is `FingerprintPairDataset`, which returns `(image1, image2, label)` for training and evaluation.

### `models/fingerprint_cnn.py`

Defines `FingerprintCNN`, the convolutional encoder. It takes one grayscale fingerprint image and converts it into a 64-value embedding vector. The output embedding is normalized so distance comparisons are more stable.

### `models/siamese_network.py`

Defines `SiameseNetwork`, which uses the same `FingerprintCNN` encoder for two input images. It returns both embeddings and the distance between them. Smaller distances mean the model thinks the fingerprints are more similar.

### `train.py`

Trains the Siamese network. It:

- builds train and validation subject splits,
- creates random fingerprint pairs,
- trains with contrastive loss,
- reports validation loss and accuracy,
- saves the best checkpoint to `checkpoints/siamese_fingerprint.pt` by default.

### `predict.py`

Loads a trained checkpoint and compares two fingerprint images. It prints the embedding distance, threshold, and prediction:

- `same subject`
- `different subjects`

### `visualization.py`

Generates Matplotlib diagrams from a trained checkpoint. It saves images into `diagrams/`, including:

- distance histogram,
- distance scatter plot,
- similarity summary chart,
- example fingerprint pair grid.

### `inspect_data.py`

Quickly checks whether the dataset is loading correctly. It prints subject count, image count, batch tensor shapes, and sample labels.

### `test.py`

A lightweight smoke test. It confirms that the dataset, CNN encoder, Siamese model, and contrastive loss can run together.

### `models/__init__.py`

Provides convenient imports for the model package.

### `requirements.txt`

Lists the Python dependencies needed to run the project.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the SOCOFing dataset under:

```text
data/SOCOFing/Real
```

The project expects `.BMP` files in that folder.

## Basic Commands

Inspect the dataset:

```bash
python inspect_data.py
```

Run the smoke test:

```bash
python test.py
```

Train the model:

```bash
python train.py
```

Train with a larger experiment:

```bash
python train.py --epochs 20 --train-pairs 20000 --validation-pairs 5000
```

Compare two fingerprints:

```bash
python predict.py path/to/image1.BMP path/to/image2.BMP --checkpoint checkpoints/siamese_fingerprint.pt
```

Generate visualizations:

```bash
python visualization.py --checkpoint checkpoints/siamese_fingerprint.pt
```

## Current Status

The project currently has a complete prototype workflow:

1. load fingerprint data,
2. create comparison pairs,
3. train a Siamese CNN,
4. save a checkpoint,
5. compare new fingerprint pairs,
6. visualize model distances.

The model architecture is intentionally simple so the full workflow is easy to understand and improve.

## Future Improvements

To improve accuracy, useful next steps include:

- Train for more epochs with more generated pairs.
- Use the altered SOCOFing fingerprints in addition to the real images.
- Tune the similarity threshold using validation data instead of relying on a fixed value.
- Add stronger image augmentation, such as small rotations, crops, contrast changes, and noise.
- Add a deeper CNN encoder or use transfer learning from a pretrained vision model.
- Track precision, recall, F1 score, and false accept / false reject rates.
- Save training history and plot loss curves over epochs.
- Build a proper test split held out from both training and validation.
- Add unit tests for filename parsing, pair generation, model shapes, and checkpoint loading.
- Add a small user interface for comparing uploaded fingerprint images.

## Notes

Large/generated files are intentionally ignored by Git:

- `data/`
- `checkpoints/`
- `diagrams/`
- virtual environments
- Python cache files

This keeps the GitHub repository focused on source code and documentation.