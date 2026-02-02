# Unit 02: Learning Process and Different Models in Neural Networks - Notes

## Introduction

Neural network learning encompasses various paradigms and algorithms that enable machines to learn from data. This unit explores the mathematical foundations and practical implementations of different learning approaches.

## Learning Paradigms

### Supervised Learning

Supervised learning involves training a model using labeled data, where both inputs and desired outputs are provided. The model learns to map inputs to outputs by minimizing prediction errors.

**Characteristics:**
- Requires labeled training data
- Learns input-output mapping
- Evaluation using test data with known labels
- Common tasks: classification, regression

### Unsupervised Learning

Unsupervised learning discovers patterns in data without labeled outputs. The model identifies hidden structures and relationships within the input data.

**Characteristics:**
- No labeled outputs required
- Discovers inherent data structure
- Common tasks: clustering, dimensionality reduction, density estimation

### Memory-Based Learning

Memory-based learning stores training examples and uses them directly for predictions. Nearest neighbor methods exemplify this approach.

**Techniques:**
- k-Nearest Neighbors (k-NN)
- Locally Weighted Regression
- Radial Basis Functions

## Hebbian Learning

### Fundamental Principle

Hebbian learning is based on the principle: "Neurons that fire together, wire together." The connection strength between neurons increases when they are simultaneously active.

**Basic Hebbian Rule:**
Δw_ij = η * x_i * y_j

Where:
- w_ij: weight from neuron i to neuron j
- η: learning rate
- x_i: input activation
- y_j: output activation

### Mathematical Models and Modifications

**Covariance Rule:**
Incorporates mean activity levels to stabilize learning.

**Oja's Rule:**
Normalizes weights to prevent unbounded growth.

**BCM Theory:**
Introduces a threshold that adapts based on neuron activity history.

## Competitive Learning

Competitive learning involves neurons competing for the right to respond to input patterns. Only the "winning" neuron updates its weights.

**Characteristics:**
- Winner-take-all mechanism
- Unsupervised learning approach
- Forms basis for self-organizing maps
- Useful for clustering and feature detection

**Learning Rule:**
Only the winning neuron j* updates:
Δw_j* = η(x - w_j*)

## Error-Correction Learning

Error-correction learning adjusts weights based on the difference between desired and actual outputs.

**General Form:**
Δw = η * error * input

This principle underlies the perceptron learning algorithm and forms the foundation for backpropagation.

## Boltzmann Learning

Boltzmann learning is a stochastic learning algorithm based on statistical mechanics principles.

**Key Concepts:**
- Energy-based models
- Probabilistic neuron activation
- Simulated annealing for optimization
- Applicable to Boltzmann machines

## Learning Tasks

### Pattern Association

Associating input patterns with output patterns. Types include:
- Auto-association: Recalling complete patterns from partial inputs
- Hetero-association: Mapping between different pattern sets

### Pattern Recognition and Function Approximation

- **Pattern Recognition:** Classifying inputs into predefined categories
- **Function Approximation:** Learning continuous mappings from inputs to outputs

### Control and Filtering

Neural networks can learn control policies and filter signals:
- Adaptive control systems
- Noise reduction
- Signal prediction

### Beamforming

Spatial filtering technique using antenna arrays, where neural networks optimize beam patterns for signal reception.

## Statistical Learning Theory

### Statistical Nature of Learning

Learning from finite data involves statistical uncertainty. Key considerations:
- Generalization from samples to population
- Bias-variance tradeoff
- Overfitting and underfitting

### Probably Approximately Correct (PAC) Model

PAC learning provides theoretical guarantees on learning performance:
- With high probability, learned hypothesis is approximately correct
- Relates sample size to accuracy and confidence

### Adaptive Filtering

Adaptive filters adjust parameters in response to changing environments:
- LMS (Least Mean Squares) algorithm
- RLS (Recursive Least Squares) algorithm

## Optimization Techniques

### Unconstrained Optimization

Finding optimal parameters without constraints:
- Gradient descent
- Conjugate gradient methods
- Quasi-Newton methods

### Linear Least-Squares Filters

Minimize squared error for linear systems:
- Closed-form solution via normal equations
- Pseudoinverse for overdetermined systems

### Least-Mean-Square (LMS) Algorithm

Iterative algorithm for adaptive filtering:
- Simple implementation
- Stochastic gradient descent
- Convergence depends on learning rate

## Learning Curves and Learning Rate Annealing

### Learning Curves

Plot of error versus training iterations or data size. Indicates:
- Convergence behavior
- Underfitting or overfitting
- Need for more data or model complexity

### Learning Rate Annealing

Gradually reducing learning rate during training:
- Faster initial convergence
- Fine-tuning in later stages
- Common schedules: step decay, exponential decay, cosine annealing

## Perceptron

### Perceptron Algorithm

Single-layer neural network for binary classification:

Output: y = sign(w^T x + b)

Weight update (for misclassification):
w := w + η * y_true * x

### Perceptron Convergence Theorem

For linearly separable data, the perceptron algorithm converges in finite steps to a solution that correctly classifies all training examples.

## Multilayer Perceptron (MLP)

### MLP Concepts

- Multiple layers of neurons
- Non-linear activation functions
- Can approximate any continuous function (universal approximation theorem)
- Overcomes limitations of single-layer perceptrons

### Architecture

- Input layer: Receives features
- Hidden layers: Extract hierarchical features
- Output layer: Produces predictions

## Backpropagation Algorithm

### Overview

Backpropagation efficiently computes gradients for weight updates in multilayer networks by propagating errors backward through the network.

### Algorithm Steps

1. **Forward pass:** Compute activations layer by layer
2. **Compute output error:** Compare predictions with targets
3. **Backward pass:** Propagate errors to hidden layers
4. **Update weights:** Adjust weights using gradients

### Mathematical Foundation

Uses chain rule of calculus to compute partial derivatives of error with respect to weights.

## XOR Problem

The XOR function is not linearly separable, demonstrating the limitation of single-layer perceptrons. An MLP with one hidden layer can solve XOR:

- Input layer: 2 neurons
- Hidden layer: 2 neurons (minimum)
- Output layer: 1 neuron

This illustrates the power of hidden layers in learning non-linear decision boundaries.

## Heuristics for Backpropagation

### Improving Performance

- Initialize weights randomly but small
- Normalize inputs
- Use appropriate learning rates
- Add momentum term
- Employ batch normalization
- Use dropout for regularization

### Output Representation

- Classification: Softmax for multi-class, sigmoid for binary
- Regression: Linear activation for output layer

## Feature Detection

Neural networks automatically learn feature detectors in hidden layers. Early layers detect simple features (edges, textures), while deeper layers detect complex patterns (objects, faces).

## Hessian Matrix

The Hessian matrix contains second-order partial derivatives of the error function:
- Provides curvature information
- Used in advanced optimization methods
- Helps analyze critical points

## Generalization and Function Approximation

### Generalization

Ability of a model to perform well on unseen data. Factors affecting generalization:
- Model complexity
- Training data size
- Regularization

### Universal Approximation Theorem

An MLP with one hidden layer and sufficient neurons can approximate any continuous function on a compact domain to arbitrary accuracy.

## Cross-Validation

Technique for assessing model generalization:
- k-fold cross-validation: Divide data into k subsets
- Train on k-1 subsets, validate on remaining subset
- Repeat k times, average results

## Network Pruning

Removing unnecessary weights or neurons to:
- Reduce model complexity
- Improve generalization
- Decrease computational requirements

**Techniques:**
- Magnitude-based pruning
- Sensitivity analysis
- Regularization (L1, L2)

## Regularization Theory and Networks

### Regularization

Adding constraints to prevent overfitting:
- L1 regularization: Promotes sparsity
- L2 regularization (weight decay): Penalizes large weights
- Dropout: Randomly deactivates neurons during training

### Regularization Networks

Combine data fitting with smoothness constraints, minimizing:
Error + λ * Complexity

Where λ controls the tradeoff between fitting and smoothness.

## Radial Basis Function (RBF) Networks

### Architecture

- Input layer
- Hidden layer with RBF activation (typically Gaussian)
- Linear output layer

### RBF vs MLP

- RBF networks use localized activation functions
- Faster training (often one-shot for output weights)
- MLP generally more compact for same approximation accuracy

### Applications

- Function approximation
- Time series prediction
- Classification

## Kernel Regression

Non-parametric method that uses kernels (similar to RBF) to weight training examples based on distance to query point. Closely related to RBF networks.

## Simulated Annealing

Global optimization technique inspired by metallurgical annealing:
- Accepts worse solutions probabilistically
- Probability decreases over time (cooling schedule)
- Helps escape local minima

## Boltzmann Machines

### Energy-Based Model

Boltzmann machines define probability distributions over configurations using an energy function:

P(v, h) ∝ exp(-E(v, h) / T)

Where v are visible units, h are hidden units, E is energy, and T is temperature.

### Deterministic Boltzmann Machine

Deterministic variant using mean-field approximation, replacing stochastic units with deterministic expectations.

## Summary

This unit covered diverse learning paradigms and neural network models, from simple perceptrons to complex Boltzmann machines. Understanding these foundational concepts and algorithms is essential for designing and training effective deep learning systems. The mathematical principles explored here underpin modern deep learning techniques covered in subsequent units.
