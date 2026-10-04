# Industrial Surface Crack Detection (CNN)

A computer vision project that detects cracks on industrial surfaces from
images using a custom-trained CNN, with a full train/validation/test
pipeline, data augmentation, and per-class evaluation.

## Problem Statement

Manual visual inspection of industrial surfaces (walls, pipelines,
structures) for cracks is time-consuming and prone to human error. This
project automates crack detection as a binary image classification task -
Crack vs No Crack - enabling faster, more consistent quality inspection.

## Dataset

A labeled surface image dataset of 40,000 images (20,000 Crack, 20,000
No Crack), organized into `Positive` (crack present) and `Negative` (no
crack) folders. The pipeline automatically splits this into train (70%),
validation (15%), and test (15%) sets, with a fixed random seed for
reproducibility.

## Approach

1. **Data Preparation** - Raw Positive/Negative images are shuffled and
   split into train/validation/test folders, each preserving the class
   structure.
2. **Augmentation** - Training images are augmented (rotation, zoom, shift,
   horizontal flip) to improve generalization on a relatively small dataset.
3. **CNN Architecture** - A 4-block CNN (32->64->128->256 filters) with
   BatchNormalization after each convolution, followed by two Dense layers
   with Dropout, ending in a sigmoid output for binary classification.
4. **Training Callbacks** - EarlyStopping (restores best weights),
   ModelCheckpoint (saves the best validation-accuracy model), and
   ReduceLROnPlateau (shrinks learning rate on plateau) work together for
   stable, well-regularized training.
5. **Evaluation** - Confusion matrix and per-class classification report
   (precision/recall/F1) on the held-out test set, plus a single-image
   prediction demo.

## Results

Evaluated on a held-out test set of 6,000 images (3,000 per class, never
seen during training or validation):

| Metric | Score |
|---|---|
| Test Accuracy | 99.95% |
| Test Loss | 0.0032 |
| Precision (both classes) | 1.00 |
| Recall (both classes) | 1.00 |
| F1-score (both classes) | 1.00 |

**Confusion Matrix:**

|  | Predicted Crack | Predicted NoCrack |
|---|---|---|
| **Actual Crack** | 2998 | 2 |
| **Actual NoCrack** | 1 | 2999 |

Only 3 misclassifications out of 6,000 test images. The crack/no-crack
distinction in this dataset is visually quite strong (surface texture
differences are pronounced), which combined with a clean, balanced,
40,000-image dataset allows the CNN to separate the two classes almost
perfectly.

## How to Run

```bash
pip install -r requirements.txt
python surface_crack_detection.py
```

Update the `ORIGINAL_DATASET` path in the script to point to your dataset's
`Positive`/`Negative` folders before running.

## Tech Stack

- Python, TensorFlow/Keras
- Scikit-learn (confusion matrix, classification report)
- Matplotlib (training curves, sample image visualization)
