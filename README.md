# Grad-CAM Implementation

A comprehensive implementation of Grad-CAM (Gradient-weighted Class Activation Mapping) for deep learning model interpretability and visualization.

## Overview

This project implements Grad-CAM, a technique that provides visual explanations for decisions made by convolutional neural networks. It highlights the regions in an image that are most important for the model's classification decision.

## Features

- **Grad-CAM Implementation**: Complete implementation of the Grad-CAM algorithm
- **Model Interpretability**: Understand what your CNN models are "looking at"
- **Visual Explanations**: Generate heatmaps showing important image regions
- **Multiple Model Support**: Works with various CNN architectures
- **Jupyter Notebook**: Interactive implementation with examples

## Project Structure

```
grad-cam/
├── TP2_GradCAM.ipynb          # Main implementation notebook
├── TP2_images.zip              # Sample images for testing
├── TP2_images/                 # Extracted sample images
├── data/                       # Additional data files
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that:
- Generates visual explanations for CNN decisions
- Shows which parts of an image influenced the classification
- Works with any CNN architecture without retraining
- Provides interpretable heatmaps

## Requirements

- Python 3.7+
- Jupyter Notebook
- PyTorch or TensorFlow
- NumPy
- OpenCV
- Matplotlib
- PIL (Pillow)

## Installation

```bash
# Clone the repository
git clone https://github.com/Jovillios/grad-cam.git
cd grad-cam

# Install dependencies
pip install torch torchvision numpy opencv-python matplotlib pillow jupyter
```

## Usage

### 1. Open the Jupyter Notebook
```bash
jupyter notebook TP2_GradCAM.ipynb
```

### 2. Run the Implementation
The notebook contains:
- **Grad-CAM Algorithm**: Complete implementation
- **Model Loading**: Pre-trained CNN models
- **Image Processing**: Load and preprocess images
- **Heatmap Generation**: Create Grad-CAM visualizations
- **Results Analysis**: Interpret the generated heatmaps

### 3. Key Functions

```python
# Generate Grad-CAM heatmap
heatmap = generate_gradcam(model, image, target_class)

# Visualize results
visualize_gradcam(image, heatmap, prediction)
```

## How Grad-CAM Works

1. **Forward Pass**: Pass image through the CNN
2. **Gradient Computation**: Compute gradients of target class w.r.t. feature maps
3. **Weight Calculation**: Calculate importance weights for each feature map
4. **Heatmap Generation**: Weighted combination of feature maps
5. **Visualization**: Overlay heatmap on original image

## Supported Models

The implementation works with:
- **ResNet**: Various ResNet architectures
- **VGG**: VGG16, VGG19
- **AlexNet**: Classic CNN architecture
- **Custom Models**: Any CNN with convolutional layers

## Customization

You can modify the implementation to:
- Use different CNN architectures
- Apply to different image datasets
- Modify the visualization style
- Add additional interpretability techniques

## Applications

Grad-CAM is useful for:
- **Model Debugging**: Understand model failures
- **Medical Imaging**: Identify important regions in medical scans
- **Autonomous Driving**: Understand what the model focuses on
- **Research**: Validate model behavior and decisions

## Performance

- **Speed**: Real-time visualization for most models
- **Memory**: Efficient memory usage
- **Accuracy**: High-quality heatmap generation

## Contributing

Contributions are welcome! Areas for improvement:
- Additional interpretability techniques
- Performance optimizations
- Support for more model types
- Enhanced visualization options

## License

This project is open source. Please check the repository for specific licensing information.

## Citation

If you use this project in your research, please cite:
```
Grad-CAM Implementation for CNN Interpretability
Author: Jules Decaestecker
Repository: https://github.com/Jovillios/grad-cam
```

## Related Work

- **Grad-CAM Paper**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **CAM**: Class Activation Mapping
- **Guided Backpropagation**: Alternative visualization technique

## Contact

For questions or contributions, please open an issue or pull request on GitHub.