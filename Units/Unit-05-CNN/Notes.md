# Unit 05: Convolutional Neural Networks - Notes

## Introduction

Convolutional Neural Networks (CNNs) represent a specialized class of neural networks designed for processing grid-structured data, particularly images. Their architecture incorporates principles from neuroscience and exploits spatial locality to achieve exceptional performance in computer vision tasks.

## Convolutional Neural Networks

### Overview

CNNs differ from standard neural networks through specialized layers that preserve spatial relationships and reduce parameter count through weight sharing.

### Key Characteristics

- **Spatial hierarchy:** Learn features at multiple scales
- **Translation invariance:** Detect features regardless of position
- **Parameter sharing:** Same filter applied across entire image
- **Local connectivity:** Neurons connect to small regions

## What and How of ConvNets

### Architecture Components

**1. Convolutional Layers:**
Apply filters to extract features

**2. Pooling Layers:**
Reduce spatial dimensions, provide translation invariance

**3. Fully Connected Layers:**
Perform classification based on extracted features

**4. Activation Functions:**
Introduce non-linearity (typically ReLU)

### Basic Operation Flow

Input Image → Convolution → Activation → Pooling → ... → Flatten → Dense → Output

## Convolution Effectiveness

### Why Convolution Works

**1. Parameter Efficiency:**
A 3×3 filter has 9 parameters regardless of image size, while a fully connected layer would require width × height parameters per neuron.

**2. Spatial Locality:**
Related pixels are nearby. Filters exploit local patterns like edges and textures.

**3. Translation Invariance:**
Same filter detects features anywhere in the image.

**4. Hierarchical Learning:**
- Early layers: Simple features (edges, colors)
- Middle layers: Textures, patterns
- Deep layers: Complex objects, parts

## What is This Convolution and Why is it Effective?

### Mathematical Definition

Convolution is a mathematical operation that combines two functions to produce a third:

(f * g)(t) = ∫ f(τ)g(t - τ)dτ

For discrete 2D images:

(I * K)(i, j) = ΣΣ I(m, n)K(i-m, j-n)

Where:
- I: Input image
- K: Kernel/filter
- (i, j): Output position

### Effectiveness Factors

**Feature Detection:**
Different filters detect different features (edges, corners, textures).

**Weight Sharing:**
Drastically reduces parameters compared to fully connected networks.

**Spatial Hierarchy:**
Stacked convolutions build complex features from simple ones.

**Biological Inspiration:**
Mimics receptive fields in visual cortex.

## Visualization of 2D Convolution

### Example: Edge Detection

**Filter (Vertical Edge):**
```
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

**Input Image (simplified):**
```
[0  0  255  255]
[0  0  255  255]
[0  0  255  255]
```

**Convolution Output:**
High response where vertical edges exist.

### Process

1. Place filter on top-left of image
2. Element-wise multiply filter and image patch
3. Sum all products
4. Move filter (stride) and repeat
5. Result: Feature map

### Parameters

**Stride:** Step size when moving filter (typically 1 or 2)
**Padding:** Add border to input (preserve spatial dimensions or allow edge detection)

### Output Size

For input size n, filter size f, padding p, stride s:

Output size = (n + 2p - f)/s + 1

## Visualization of 3D Convolution

### Multi-Channel Input

Color images have 3 channels (RGB). Filters also have 3 channels.

**Filter Shape:** (height, width, input_channels, num_filters)

**Operation:**
- Each filter spans all input channels
- Produces one activation map
- Multiple filters produce multiple output channels

### Example

Input: 32×32×3 (RGB image)
Filter: 3×3×3
Number of filters: 64
Output: 30×30×64 (with stride 1, no padding)

Each of 64 filters detects different features across all color channels.

### Depth-wise vs Standard Convolution

**Standard Convolution:**
Filter spans all input channels

**Depthwise Separable Convolution:**
Separates spatial and channel-wise convolutions for efficiency

## Building a Model Without Max-Pooling Layers

### Motivation

Max-pooling reduces spatial resolution, potentially losing information. Alternatives:

**Strided Convolutions:**
Use stride > 1 to reduce dimensions while learning the downsampling

**Advantages:**
- Learned downsampling vs fixed pooling
- Retains more information
- More parameters but often better performance

**Disadvantages:**
- More computation
- Risk of overfitting without regularization

## How to Train a CNN on a Dataset from Ground-Up

### Steps

**1. Data Collection:**
Gather labeled images for training

**2. Data Exploration:**
Understand class distribution, image sizes, quality

**3. Preprocessing:**
- Resize to consistent dimensions
- Normalize pixel values (e.g., [0, 1] or [-1, 1])
- Convert labels to appropriate format

**4. Model Architecture:**
Design CNN with appropriate layers

**5. Compilation:**
Choose loss, optimizer, metrics

**6. Training:**
Fit model on training data with validation

**7. Evaluation:**
Test on held-out test set

**8. Iteration:**
Adjust architecture, hyperparameters based on results

## Importance of Deep Learning When Data is Limited

### Challenges with Limited Data

Small datasets lead to:
- Overfitting
- Poor generalization
- High variance

### Solutions

**1. Data Augmentation:**
Artificially expand dataset through transformations

**2. Transfer Learning:**
Use pre-trained models, fine-tune on small dataset

**3. Regularization:**
Dropout, L2 penalty to prevent overfitting

**4. Simpler Models:**
Reduce capacity to match data availability

**5. Semi-supervised/Unsupervised Pre-training:**
Leverage unlabeled data

## Data Preprocessing: Preparing the Data

### Image Preprocessing

**Normalization:**
```python
X_train = X_train.astype('float32') / 255.0
```

**Standardization:**
```python
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
```

**Resizing:**
```python
from tensorflow.keras.preprocessing.image import img_to_array, load_img
img = load_img('path/to/image.jpg', target_size=(224, 224))
```

**Label Encoding:**
```python
from tensorflow.keras.utils import to_categorical
y_train_categorical = to_categorical(y_train, num_classes=10)
```

## Making the Most of What's Available: Data Augmentation

### Concept

Generate new training samples by applying random transformations to existing images.

### Common Augmentations

- Rotation
- Width/height shift
- Horizontal/vertical flip
- Zoom
- Brightness adjustment
- Contrast adjustment

### Implementation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

datagen.fit(X_train)
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          epochs=50)
```

### Benefits

- Increases effective dataset size
- Improves generalization
- Reduces overfitting
- Model becomes robust to variations

## Using a Trained CNN (Transfer Learning)

### Concept

Leverage features learned by models trained on large datasets (e.g., ImageNet) for new tasks.

### Approaches

**Feature Extraction:**
1. Use pre-trained CNN as fixed feature extractor
2. Remove top layers
3. Add new classifier
4. Train only new layers

**Fine-tuning:**
1. Start with pre-trained weights
2. Unfreeze some layers
3. Train with low learning rate

### Implementation

```python
from tensorflow.keras.applications import VGG16

# Load pre-trained model without top layers
base_model = VGG16(weights='imagenet', 
                   include_top=False, 
                   input_shape=(224, 224, 3))

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### Benefits

- Requires less data
- Faster training
- Often better performance
- Especially effective when tasks are related

## Building a CNN, One Layer at a Time

### Example Architecture for Image Classification

```python
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### Design Principles

- Start with smaller filters (32-64)
- Increase filter count in deeper layers
- Alternate convolution and pooling
- Use ReLU activation
- Add dropout before final layers
- Match output layer to task (softmax for classification)

## Tuning the CNN

### Hyperparameters to Tune

**Architecture:**
- Number of layers
- Filters per layer
- Filter size
- Pooling strategy

**Training:**
- Learning rate
- Batch size
- Number of epochs
- Optimizer choice

**Regularization:**
- Dropout rate
- L2 penalty
- Data augmentation intensity

### Systematic Tuning

1. Establish baseline
2. Change one parameter at a time
3. Use validation set for selection
4. Consider automated methods (grid search, random search, Bayesian optimization)

## What Do Convolutional Neural Networks See?

### Layer-wise Feature Learning

**First Layers:**
- Edges (horizontal, vertical, diagonal)
- Color blobs
- Simple gradients

**Middle Layers:**
- Textures
- Patterns
- Parts of objects

**Deep Layers:**
- Whole objects
- Faces
- Scene components

### Visualization Methods

**1. Activation Maximization:**
Generate input that maximally activates a neuron

**2. Deconvolution:**
Project activations back to pixel space

**3. Gradient-based methods:**
Saliency maps, Grad-CAM

## Seeing the Intermediate Layers

### Extracting Activations

```python
# Create model that outputs intermediate layer activations
layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# Get activations for an input
activations = activation_model.predict(sample_input)
```

### Visualization

```python
import matplotlib.pyplot as plt

# Visualize feature maps from first conv layer
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
```

### Insights

- Early layers: Localized, interpretable features
- Deep layers: Abstract, distributed representations
- Dead filters: Neurons that never activate (potential inefficiency)

## Visualizing the Filters Themselves

### Filter Visualization

Display learned filter weights:

```python
first_conv_layer = model.layers[0]
filters, biases = first_conv_layer.get_weights()

# Normalize for visualization
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Display filters
n_filters = 32
for i in range(n_filters):
    plt.subplot(4, 8, i+1)
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.show()
```

### Interpretation

- First layer filters often resemble edge detectors
- Deeper filters are harder to interpret visually
- Random-looking filters may indicate poor training

## Looking at Heat Maps: How Filters Seek Details

### Grad-CAM (Gradient-weighted Class Activation Mapping)

Produces heat map showing which regions of image are important for prediction.

### Implementation Concept

1. Get gradients of class score with respect to final conv layer
2. Weight feature maps by gradients
3. Combine weighted maps
4. Apply ReLU (focus on positive influence)
5. Upsample to image size
6. Overlay on original image

### Applications

- Model interpretability
- Debugging incorrect predictions
- Verifying model focuses on relevant features
- Building trust in model decisions

## Summary

Convolutional Neural Networks leverage the spatial structure of images through localized filters, weight sharing, and hierarchical feature learning. Key concepts include:

- Convolution operations detect local patterns efficiently
- Pooling reduces dimensions and provides invariance
- Data augmentation and transfer learning address limited data scenarios
- Visualization techniques reveal what networks learn at different depths
- Systematic architecture design and tuning improve performance

CNNs have revolutionized computer vision, enabling breakthroughs in image classification, object detection, segmentation, and many other applications. Understanding their operation and proper training is essential for modern deep learning practitioners.
