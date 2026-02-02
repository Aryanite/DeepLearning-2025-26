# Unit 04: Diving to the Depths of Deep Learning - Notes

## Introduction

This unit transitions from theoretical foundations to practical implementation of deep learning systems. It covers the essential components, design choices, and methodologies for building effective neural networks.

## Deep Learning Depths

### Depth in Neural Networks

Network depth refers to the number of layers between input and output. Deeper networks can:
- Learn more complex hierarchical representations
- Achieve better performance on complex tasks
- Require more data and computational resources

### Empirical Observations

- Very shallow networks: Limited expressiveness
- Moderate depth: Balances performance and trainability
- Very deep networks: Prone to optimization difficulties (addressed by skip connections, normalization)

## Model: The Molecules of Deep Learning

### Basic Building Blocks

**Layers:**
- Dense (Fully Connected)
- Convolutional
- Recurrent
- Normalization
- Dropout
- Pooling

**Connections:**
- Sequential: Layer outputs feed to next layer
- Skip/Residual: Connections bypass layers
- Branching: Parallel paths

**Parameters:**
- Weights
- Biases
- Hyperparameters (learning rate, batch size)

### Model Design Principles

1. Start simple, add complexity as needed
2. Match architecture to problem domain
3. Consider computational constraints
4. Balance expressiveness with generalization

## Loss Functions in Neural Networks

### Purpose

Loss functions quantify the difference between predictions and true values, providing the objective for optimization.

### Classification Loss Functions

**Binary Cross-Entropy:**
For binary classification:

L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

**Categorical Cross-Entropy:**
For multi-class classification:

L = -Σ y_i * log(ŷ_i)

Where y is one-hot encoded true label.

**Sparse Categorical Cross-Entropy:**
Same as categorical but accepts integer labels instead of one-hot encoding.

### Regression Loss Functions

**Mean Squared Error (MSE):**

L = (1/n) * Σ(y - ŷ)²

Common for regression, sensitive to outliers.

**Mean Absolute Error (MAE):**

L = (1/n) * Σ|y - ŷ|

More robust to outliers than MSE.

**Huber Loss:**

Combines MSE and MAE, providing robustness and smoothness.

### Selection Criteria

- Classification: Cross-entropy (interpretable as log-likelihood)
- Regression: MSE for smooth gradients, MAE for robustness
- Custom losses for specialized tasks

## Optimizers in Neural Networks

### Gradient Descent Variants

**Batch Gradient Descent:**
- Uses entire dataset for each update
- Stable but slow for large datasets

**Stochastic Gradient Descent (SGD):**
- Uses single sample for each update
- Fast but noisy updates

**Mini-batch Gradient Descent:**
- Uses small batches
- Balances speed and stability
- Most common in practice

### Advanced Optimizers

**Momentum:**
Accumulates velocity in gradient direction:

v_t = β*v_{t-1} + ∇L
θ_t = θ_{t-1} - η*v_t

Accelerates in consistent directions, dampens oscillations.

**RMSprop:**
Adapts learning rate per parameter:

s_t = β*s_{t-1} + (1-β)*(∇L)²
θ_t = θ_{t-1} - η*∇L/√(s_t + ε)

**Adam (Adaptive Moment Estimation):**
Combines momentum and RMSprop:

m_t = β₁*m_{t-1} + (1-β₁)*∇L
v_t = β₂*v_{t-1} + (1-β₂)*(∇L)²
θ_t = θ_{t-1} - η*m_t/(√v_t + ε)

**AdamW:**
Adam with decoupled weight decay, often better generalization.

### Choosing an Optimizer

- **Adam:** Good default choice, works well across problems
- **SGD with momentum:** Often better final performance with tuning
- **RMSprop:** Good for RNNs
- Experiment to find best for specific task

## Activation Functions

### Purpose

Activation functions introduce non-linearity, enabling networks to learn complex patterns.

### Common Activation Functions

**ReLU (Rectified Linear Unit):**

f(x) = max(0, x)

Advantages:
- Simple and fast
- Mitigates vanishing gradient
- Sparse activation

Disadvantages:
- Dead neurons (always output 0)

**Leaky ReLU:**

f(x) = x if x > 0 else α*x (α small, e.g., 0.01)

Addresses dead neuron problem.

**ELU (Exponential Linear Unit):**

f(x) = x if x > 0 else α*(e^x - 1)

Smooth, can produce negative outputs.

**Sigmoid:**

f(x) = 1/(1 + e^(-x))

Outputs in (0, 1), used for binary classification output layer.
Suffers from vanishing gradient in hidden layers.

**Tanh:**

f(x) = (e^x - e^(-x))/(e^x + e^(-x))

Outputs in (-1, 1), zero-centered.
Similar vanishing gradient issues as sigmoid.

**Softmax:**

f(x_i) = e^(x_i) / Σ e^(x_j)

Converts logits to probability distribution.
Used for multi-class classification output layer.

**Swish/SiLU:**

f(x) = x * sigmoid(x)

Smooth, non-monotonic, performs well in deep networks.

### Selection Guidelines

- Hidden layers: ReLU or variants (Leaky ReLU, ELU)
- Binary classification output: Sigmoid
- Multi-class classification output: Softmax
- Regression output: Linear (no activation)

## Finding the Perfect Fit

### Bias-Variance Tradeoff

- **High bias (underfitting):** Model too simple, poor training and validation performance
- **High variance (overfitting):** Model too complex, good training but poor validation performance
- **Good fit:** Balanced, good performance on both training and validation data

### Strategies

**Addressing Underfitting:**
- Increase model complexity
- Add features
- Train longer
- Reduce regularization

**Addressing Overfitting:**
- Collect more data
- Reduce model complexity
- Add regularization (L1, L2, dropout)
- Data augmentation
- Early stopping

## Running Deep Learning Algorithms: The Frameworks

### Popular Frameworks

**TensorFlow/Keras:**
- High-level API (Keras) and low-level control (TensorFlow)
- Production-ready deployment tools
- Extensive ecosystem

**PyTorch:**
- Dynamic computational graphs
- Pythonic and intuitive
- Popular in research

**JAX:**
- High-performance automatic differentiation
- Functional programming approach

### Typical Workflow

1. Import libraries
2. Load and preprocess data
3. Define model architecture
4. Compile model (specify loss, optimizer, metrics)
5. Train model
6. Evaluate model
7. Make predictions

## Data Preparation and Label Preparation

### Data Preparation Steps

**1. Collection:**
Gather relevant data for the task.

**2. Exploration:**
Understand data distribution, detect anomalies.

**3. Cleaning:**
- Handle missing values
- Remove duplicates
- Correct errors

**4. Normalization/Standardization:**
Scale features to similar ranges:
- Min-max scaling: x' = (x - min)/(max - min)
- Standardization: x' = (x - μ)/σ

**5. Splitting:**
Divide into training, validation, and test sets (e.g., 70%-15%-15%).

### Label Preparation

**Classification:**
- Encode categorical labels (one-hot encoding, label encoding)
- Balance classes if necessary (oversampling, undersampling)

**Regression:**
- Ensure labels are on appropriate scale
- Consider log transformation for highly skewed targets

## Examples of Neural Networks at Work

### Classification Example: Image Recognition

**Architecture:**
- Input: Flattened pixel values
- Hidden layers: Dense layers with ReLU
- Output: Softmax layer with 10 units (for 10 classes)

**Training:**
- Loss: Categorical cross-entropy
- Optimizer: Adam
- Metrics: Accuracy

### Regression Example: House Price Prediction

Covered in detail in separate section below.

## Constructing the Network

### Using Keras (TensorFlow)

```python
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Regression output
])
```

### Compilation

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

### Training

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
```

## ReLU Activation

### Why ReLU?

ReLU has become the default activation for hidden layers due to:
- Computational efficiency
- Mitigation of vanishing gradient problem
- Sparse representations
- Empirically strong performance

### Implementation

```python
layers.Dense(64, activation='relu')
# Or explicitly
layers.Dense(64)
layers.ReLU()
```

## Approach Validation

### Holdout Validation

Split data into training and validation sets. Simple but validation performance can be sensitive to split.

### Cross-Validation

More robust, especially with limited data. K-fold cross-validation uses each data portion for validation once.

### Stratified Sampling

For classification, ensure class proportions are similar in all splits.

## Plotting Loss from Validation and Training

### Purpose

Visualizing training and validation loss over epochs helps diagnose:
- Convergence
- Overfitting (validation loss increases while training loss decreases)
- Underfitting (both losses remain high)

### Implementation

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Interpretation

- Both decreasing: Good progress
- Validation loss increases: Overfitting, consider early stopping
- Both high: Underfitting, increase model capacity or train longer

## What Experiments Do We Run Next?

### Systematic Experimentation

1. **Baseline:** Simple model, establish performance
2. **Increase capacity:** More layers/neurons
3. **Regularization:** Dropout, L2 penalty
4. **Data augmentation:** If applicable
5. **Architecture variations:** Different activation functions, layer types
6. **Hyperparameter tuning:** Learning rate, batch size, optimizer
7. **Ensemble methods:** Combine multiple models

### Tracking Experiments

Record:
- Model architecture
- Hyperparameters
- Training/validation performance
- Training time

Tools: TensorBoard, Weights & Biases, MLflow

## Regression Example: House Price Prediction

### Problem Statement

Predict house prices based on features like area, number of bedrooms, location, etc.

### Processing the Data

**1. Load Data:**
```python
import pandas as pd
data = pd.read_csv('housing.csv')
```

**2. Explore:**
```python
data.describe()
data.info()
```

**3. Handle Missing Values:**
```python
data = data.fillna(data.median())
```

**4. Feature Engineering:**
Create new features, encode categorical variables.

**5. Normalization:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**6. Split Data:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### Building the Network

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

### Training

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
```

### Evaluation

```python
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')
```

## K-Fold Approach for Validating Algorithm

### Concept

Divide data into k folds. Train k models, each time using k-1 folds for training and 1 fold for validation. Average results for robust performance estimate.

### Advantages

- More reliable performance estimate
- Uses all data for both training and validation
- Reduces variance in results

### Disadvantages

- k times more computational cost
- More complex implementation

## K-Fold Approach: In Code

```python
from sklearn.model_selection import KFold
import numpy as np

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

val_scores = []

for train_idx, val_idx in kfold.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Build model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    model.fit(X_train_fold, y_train_fold, 
              epochs=100, batch_size=32, verbose=0)
    
    # Validate
    val_loss, val_mae = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    val_scores.append(val_mae)

print(f'Average validation MAE: {np.mean(val_scores):.2f} (+/- {np.std(val_scores):.2f})')
```

## Summary

This unit covered the core practical aspects of deep learning, from model components to end-to-end implementation. Key points:

- Loss functions measure prediction error and guide optimization
- Optimizers like Adam efficiently update model parameters
- Activation functions like ReLU introduce non-linearity
- Data preparation is critical for model performance
- Validation strategies like k-fold cross-validation provide robust performance estimates
- Systematic experimentation and monitoring training/validation curves guide model improvement

With these foundations, students can build, train, and evaluate neural networks for real-world classification and regression tasks.
