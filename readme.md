# Digit Classifier Project

A scalable and configurable handwritten digit classification system built with PyTorch.

## Overview

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. It's designed to be:

- **Educational**: Clear architecture with explanatory comments
- **Configurable**: Command-line options to customize training parameters
- **Reproducible**: Shell script for consistent experiment execution
- **Scalable**: GPU acceleration when available

## Dataset

The MNIST dataset is automatically downloaded when you run the script. It consists of:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images of handwritten digits (0-9)

## Project Structure

```
.
├── digit_classifier.py  # Main Python code with model and training logic
├── train.sh            # Shell script for running experiments
├── data/               # Downloaded dataset (created on first run)
└── experiments/        # Experiment results (created by train.sh)
    └── run_TIMESTAMP/  # Results from each training run
        ├── full_model/
        ├── no_dropout/
        ├── no_batchnorm/
        ├── baseline/
        └── summary.md  # Performance comparison
```

## Features

- Convolutional Neural Network architecture
- Batch normalization and dropout for regularization
- Command-line configuration options
- Training visualizations (training curves, prediction samples)
- Comprehensive experiments with multiple model configurations
- GPU acceleration (when available)

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy

## Usage

### Basic Training

```bash
python digit_classifier.py
```

### With Custom Parameters

```bash
python digit_classifier.py --batch-size 128 --epochs 10 --lr 0.001 --save-model
```

### Run Experiments

To compare different model configurations:

```bash
bash train.sh
```

This will train multiple model variations and save the results to the `experiments` directory.

## Command Line Options

- `--batch-size`: Input batch size for training (default: 64)
- `--epochs`: Number of epochs to train (default: 5)
- `--lr`: Learning rate (default: 0.01)
- `--no-cuda`: Disables CUDA training
- `--no-dropout`: Disables dropout regularization
- `--no-batch-norm`: Disables batch normalization
- `--save-model`: Save the trained model

## Model Architecture

```
DigitClassifier(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
  (bn1): BatchNorm2d(32)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
  (bn2): BatchNorm2d(64)
  (fc1): Linear(in_features=3136, out_features=128)
  (bn3): BatchNorm1d(128)
  (dropout): Dropout(p=0.5)
  (fc2): Linear(in_features=128, out_features=10)
)
```

## Results

After running the experiments, you'll find comparison metrics in `experiments/run_TIMESTAMP/summary.md` that show:

- Training time for each configuration
- Final accuracy metrics
- Loss and accuracy curves

## Extending the Project

1. Try different model architectures
2. Implement data augmentation
3. Add learning rate scheduling
4. Apply transfer learning techniques
5. Try different optimizers (Adam, RMSprop)