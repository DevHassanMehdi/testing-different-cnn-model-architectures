
# CNN Model Training on CIFAR-10

## Overview

This project trains multiple Convolutional Neural Network (CNN) models on the CIFAR-10 dataset to compare their performance. The models vary in architecture, incorporating techniques such as dropout, batch normalization, and global average pooling.

## Installation

Ensure you have the necessary dependencies installed:

```bash
pip install tensorflow matplotlib numpy pandas
```

## Dataset

The CIFAR-10 dataset is used, consisting of 60,000 images (32x32 pixels, 10 classes). The dataset is preprocessed by normalizing pixel values and converting labels to categorical format.

## Models Implemented

- **Basic CNN**: A simple CNN with three convolutional layers.
- **CNN with Dropout**: Adds dropout layers to prevent overfitting.
- **Deeper CNN**: Increases the number of convolutional layers.
- **CNN with Batch Normalization**: Uses batch normalization for stable training.
- **CNN with Global Average Pooling**: Replaces fully connected layers with global average pooling.
- **VGG-like CNN**: Inspired by VGG architecture with deeper layers.

## Training

Each model is trained using the Adam optimizer with categorical cross-entropy loss for 10 epochs:

```python
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

## Evaluation

- The model architectures are summarized using `model.summary()`.
- Training and validation accuracy/loss are plotted.
- A comparison table summarizes the models' performance.

## Results

A summary table is created using pandas:

```python
import pandas as pd
summary = pd.DataFrame({
    'Model': models,
    'Train Accuracy': train_accuracies,
    'Val Accuracy': val_accuracies,
    'Train Loss': train_losses,
    'Val Loss': val_losses
})
```

## Conclusion

By comparing different architectures, we gain insights into how different design choices impact model performance on CIFAR-10.
