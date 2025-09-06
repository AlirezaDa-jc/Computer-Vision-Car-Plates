# Computer-Vision-Car-Plates Documentation

## Project Overview
This project provides a complete pipeline for **license plate processing** using computer vision. The pipeline supports:
- Labeling and masking car license plates.
- Extracting plate regions from images.
- Blurring license plates.
- Training a CNN-based regression model to predict plate corner coordinates.
- Data augmentation to enhance dataset diversity and robustness.

Implemented in Python using **OpenCV**, **Keras/TensorFlow**, **NumPy**, and **imgaug**.

## Directory Structure
```
Computer-Vision-Car-Plates/
│
├─ classification/          # Data augmentation scripts
│  └─ augmentation.py       # Generates augmented synthetic license plate images
├─ cnn_dataset/             # CNN training images and labels
├─ images/                  # Raw input images
├─ labels/                  # YOLO format label files
├─ output/                  # Masked license plates
├─ output_blur/             # Blurred license plates
├─ output_extract/          # Extracted license plate images
├─ main_blur.py             # Blurring pipeline
├─ main_cnn.py              # CNN training pipeline
├─ main_extract.py          # Plate extraction pipeline
├─ main_masking.py          # Masking pipeline
├─ blurring.py              # Blur helper functions
├─ extract.py               # Plate extraction helper functions
├─ loader.py                # Dataset loader and augmentation
├─ masking.py               # Masking helper functions
├─ regressor.py             # CNN model architecture
└─ train.py                 # Model training script
```

## Module Descriptions

### classification/augmentation.py
- Applies affine transforms, rotation, translation, scaling, Gaussian noise, blurring, brightness/contrast adjustments, and JPEG compression to synthetic license plate images.
- Functions:
  - `augment_image(image)`: Applies augmentation sequence to a single image.
  - `generate_dataset(num_samples, save_path)`: Generates a dataset of synthetic augmented images.

### blurring.py
- Blurs license plate regions in images.
- Functions:
  - `blur(image, points)`: Applies Gaussian blur to the plate region defined by `points`.

### extract.py
- Extracts license plate regions from images.
- Functions:
  - `extract(image, points)`: Warps the plate region to a standard rectangle.

### loader.py
- Loads images and labels, applies augmentations, and normalizes the data.
- Functions:
  - `augment(image, points)`: Random rotation, translation, and scaling.
  - `resize(image, points, size)`: Resize image and adjust points.
  - `load(dataset_dir)`: Loads images and corresponding points from dataset.

### masking.py
- Masks license plate regions with a cover image.
- Functions:
  - `mask(image, points, cover)`: Warps cover image onto plate region.

### main_blur.py
- Processes images and applies blurring using `blurring.py`.

### main_extract.py
- Extracts license plate images from the dataset using `extract.py`.

### main_masking.py
- Applies masking to plate regions using `masking.py`.

### main_cnn.py
- Builds and trains a regression CNN to predict plate corner coordinates.
- Uses `loader.py` for dataset preparation and `regressor.py` for model architecture.

### regressor.py
- Defines CNN architecture for predicting plate corner coordinates.
- Uses multiple Conv2D layers with BatchNorm, LeakyReLU, and Dense layers.

### train.py
- High-level training script that loads data, splits into train/validation, and trains the regression model.

## Usage
1. **Prepare dataset**: Place images in `images/` and labels in `labels/`.
2. **Masking**: Run `main_masking.py`.
3. **Blurring**: Run `main_blur.py`.
4. **Extracting**: Run `main_extract.py`.
5. **Train CNN**: Run `main_cnn.py` or `train.py` for training the regression model.

## Requirements
- Python >= 3.8
- OpenCV
- NumPy
- TensorFlow/Keras
- imgaug
- scikit-learn

