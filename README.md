# Grad-CAM Practical Assignment (TP2)

An educational implementation of Grad-CAM (Gradient-weighted Class Activation Mapping) for understanding CNN model interpretability and visualization techniques.

## Overview

This is a practical assignment (TP2) that demonstrates how to implement and visualize Grad-CAM on convolutional neural networks. The project focuses on understanding how CNNs make classification decisions by highlighting the regions in an image that are most important for the model's prediction.

## Learning Objectives

- **Understand Grad-CAM**: Learn how gradient-based localization works
- **CNN Interpretability**: Explore what convolutional neural networks "see"
- **Visual Explanations**: Generate and interpret heatmaps showing important image regions
- **DenseNet Architecture**: Work with pre-trained DenseNet-121 model
- **Practical Implementation**: Hands-on coding experience with PyTorch hooks

## Project Structure

```
grad-cam/
├── TP2_GradCAM.ipynb          # Main assignment notebook with implementation
├── TP2_images.zip             # Sample images archive for testing
├── TP2_images/                # Extracted sample images (21 ImageNet images)
├── data/                      # Data directory
│   └── TP2_images/            # Nested image data structure
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that:
- Generates visual explanations for CNN decisions
- Shows which parts of an image influenced the classification
- Works with any CNN architecture without retraining
- Provides interpretable heatmaps

## Requirements

This assignment requires:
- **Python 3.7+**
- **Jupyter Notebook**
- **PyTorch** (with torchvision)
- **NumPy**
- **Matplotlib**
- **PIL (Pillow)**

## Installation

```bash
# Clone the repository
git clone https://github.com/Jovillios/grad-cam.git
cd grad-cam

# Install dependencies
pip install torch torchvision numpy matplotlib pillow jupyter
```

## Usage

### 1. Start Jupyter Notebook
```bash
jupyter notebook TP2_GradCAM.ipynb
```

### 2. Follow the Assignment Structure
The notebook is organized into several parts:
- **Part 1**: Grad-CAM implementation using PyTorch hooks
- **Part 2**: Testing on 1-3 images with analysis
- **Part 3**: Experimenting with different convolutional layers  
- **Part 4**: Special case analysis on specific image
- **Part 5**: Understanding GradCAM contributions from the research paper
- **Bonus**: Comparison between DenseNet and ResNet architectures

### 3. Key Components
- **DenseNet-121**: Pre-trained model on ImageNet (1000 classes)
- **Image preprocessing**: Resize to 224x224, normalization
- **Hook functions**: Capture gradients during backward pass
- **Visualization**: Generate and display heatmaps using matplotlib

## How Grad-CAM Works

The assignment demonstrates these key steps:

1. **Forward Pass**: Pass image through the DenseNet-121 CNN
2. **Hook Registration**: Register backward hooks to capture gradients
3. **Gradient Computation**: Compute gradients of target class with respect to feature maps  
4. **Weight Calculation**: Calculate importance weights for each feature map channel
5. **Heatmap Generation**: Weighted combination of feature maps with ReLU activation
6. **Visualization**: Overlay heatmap on original image using matplotlib

## Assignment Scope

This educational project focuses on:
- **Single Model**: DenseNet-121 pre-trained on ImageNet
- **Sample Dataset**: 21 provided ImageNet images
- **Educational Purpose**: Understanding CNN interpretability concepts
- **Implementation Practice**: Working with PyTorch hooks and gradients

## Learning Outcomes

Upon completing this assignment, you will understand:
- **CNN Interpretability**: How to visualize what neural networks focus on
- **PyTorch Hooks**: Practical use of forward and backward hooks
- **Gradient-based Methods**: How gradients can provide model insights
- **DenseNet Architecture**: Working with dense connectivity patterns  
- **Research Implementation**: Translating academic papers into working code

## Technical Details

- **Model**: DenseNet-121 pre-trained on ImageNet (1000 classes)
- **Input Size**: 224x224 RGB images
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Framework**: PyTorch with torchvision transforms

## Educational Context

This is practical work designed for learning computer vision and deep learning interpretability. The implementation includes guided sections and questions to deepen understanding of:
- How convolutional layers capture visual features
- The relationship between gradients and visual attention
- Differences between various CNN architectures
- Critical analysis of model behavior

## Citation

If you use this educational material in academic work, please reference:
```
Grad-CAM Practical Assignment Implementation
Author: Jules Decaestecker
Repository: https://github.com/Jovillios/grad-cam
```

## Related Work

- **Original Paper**: Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017
- **DenseNet Paper**: Huang, G., et al. "Densely Connected Convolutional Networks." CVPR 2017
- **CAM**: Class Activation Mapping (Zhou et al.)

## Contact

For questions about this assignment, please open an issue or contact via GitHub.