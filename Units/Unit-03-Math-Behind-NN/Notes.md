# Unit 03: The Math Behind Neural Networks - Notes

## Introduction

Mathematics forms the backbone of neural networks and deep learning. This unit explores the essential mathematical concepts, from basic linear algebra to advanced tensor operations and gradient calculations, that enable the functioning of deep learning systems.

## How Does a Neural Network Look Like?

### Mathematical Representation

A neural network can be viewed as a composition of functions:

y = f_n(f_{n-1}(...f_2(f_1(x))))

Where each f_i represents a layer transformation combining linear and non-linear operations.

### Layer Representation

For a single layer:

z = Wx + b
a = σ(z)

Where:
- x: input vector
- W: weight matrix
- b: bias vector
- σ: activation function
- z: pre-activation
- a: activation (output)

## The Matrix Magic

### Why Matrices?

Matrices enable efficient representation and computation of neural network operations:
- Parallel processing of multiple inputs
- Compact notation
- Hardware optimization (GPUs excel at matrix operations)

### Weight Matrices

In a layer connecting n input neurons to m output neurons:
- Weight matrix W has dimensions m × n
- Each row represents weights for one output neuron
- Matrix multiplication efficiently computes all neuron outputs simultaneously

### Example

For 3 inputs to 2 outputs:

```
W = [w11  w12  w13]
    [w21  w22  w23]

x = [x1]
    [x2]
    [x3]

z = Wx = [w11*x1 + w12*x2 + w13*x3]
         [w21*x1 + w22*x2 + w23*x3]
```

## Visualizing Deep Learning

### Computational Graphs

Neural networks can be visualized as directed acyclic graphs where:
- Nodes represent operations or variables
- Edges represent data flow
- Forward pass: left to right
- Backward pass (backpropagation): right to left

### Layer Visualization

Each layer transforms data:
- Input: Feature space
- Hidden layers: Learned representations
- Output: Prediction space

Dimensionality changes through layers reflect learned abstractions.

## The Elephant in the Room

### Curse of Dimensionality

As dimensionality increases:
- Data becomes sparse in high-dimensional spaces
- Distance metrics become less meaningful
- Computational complexity grows exponentially

Deep learning partially addresses this through:
- Automatic feature learning
- Hierarchical representations
- Dimensionality reduction in hidden layers

### Interpretability Challenges

While mathematically precise, the complexity of deep networks makes interpretation difficult. Understanding emerges through:
- Visualization techniques
- Ablation studies
- Mathematical analysis of simplified models

## Programmatic Expression of Deep Learning Math Constructs

### Tensors in Code

Tensors are multi-dimensional arrays:
- 0D tensor: Scalar
- 1D tensor: Vector
- 2D tensor: Matrix
- nD tensor: Higher-dimensional array

**Python Example (using NumPy):**
```python
import numpy as np

scalar = np.array(5)           # 0D
vector = np.array([1, 2, 3])   # 1D
matrix = np.array([[1, 2],     # 2D
                   [3, 4]])
tensor_3d = np.zeros((2, 3, 4)) # 3D
```

### Matrix Multiplication

```python
# Layer computation
z = np.dot(W, x) + b
a = activation_function(z)
```

## Operations with Tensors

### Element-wise Operations

Apply operation to corresponding elements:
- Addition: A + B
- Multiplication: A * B (Hadamard product)
- Activation functions: σ(A)

### Aggregation Operations

- Sum: np.sum(A)
- Mean: np.mean(A)
- Max: np.max(A)
- Along axes: np.sum(A, axis=0)

### Reshaping

```python
x = np.array([1, 2, 3, 4, 5, 6])
x_reshaped = x.reshape(2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]
```

### Transposition

```python
A = np.array([[1, 2], [3, 4]])
A_T = A.T  # or np.transpose(A)
# [[1, 3],
#  [2, 4]]
```

## Array Broadcasting

Broadcasting enables operations on arrays of different shapes by automatically expanding dimensions.

### Broadcasting Rules

1. If arrays differ in number of dimensions, prepend 1s to the shape of the smaller array
2. Arrays are compatible if dimensions are equal or one is 1
3. Broadcast along dimensions of size 1

### Examples

```python
# Broadcasting a scalar
A = np.array([[1, 2], [3, 4]])
result = A + 5  # Broadcasts 5 to match A's shape
# [[6, 7], [8, 9]]

# Broadcasting a vector
A = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])
result = A + b  # Broadcasts b to each row
# [[11, 22, 33],
#  [14, 25, 36]]
```

### Application in Neural Networks

Bias addition uses broadcasting:
```python
z = np.dot(W, x) + b  # b broadcasts to match z's shape
```

## Scalar Product and Inner Product of Tensors

### Dot Product (Vectors)

For vectors u and v:

u · v = Σ u_i * v_i

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot_product = np.dot(u, v)  # 1*4 + 2*5 + 3*6 = 32
```

### Matrix Multiplication

For matrices A (m×n) and B (n×p):

(AB)_ij = Σ_k A_ik * B_kj

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
# [[19, 22],
#  [43, 50]]
```

### Tensor Contraction

Generalizes dot product to higher-dimensional tensors:
```python
# For 3D tensors
result = np.tensordot(A, B, axes=([1], [0]))
```

## Morphing Shapes of Tensors

### Flatten

Convert multi-dimensional tensor to 1D:
```python
A = np.array([[1, 2], [3, 4]])
flattened = A.flatten()  # [1, 2, 3, 4]
```

### Reshape

Change dimensions while preserving total elements:
```python
A = np.arange(12)
# Reshape to 3x4
B = A.reshape(3, 4)
# Reshape to 2x2x3
C = A.reshape(2, 2, 3)
```

### Expand Dimensions

Add axes:
```python
x = np.array([1, 2, 3])  # Shape: (3,)
x_expanded = np.expand_dims(x, axis=0)  # Shape: (1, 3)
# Or using indexing
x_expanded = x[np.newaxis, :]
```

### Squeeze

Remove dimensions of size 1:
```python
A = np.array([[[1, 2, 3]]])  # Shape: (1, 1, 3)
A_squeezed = np.squeeze(A)    # Shape: (3,)
```

### Applications

Shape manipulation is crucial for:
- Matching dimensions for operations
- Preparing data for layers
- Interpreting outputs

## Gradient Calculation

### Importance

Gradients indicate direction of steepest ascent. For optimization (minimization), we move opposite to the gradient.

### Partial Derivatives

For function f(x, y):

∂f/∂x: Rate of change of f with respect to x (holding y constant)

### Gradient Vector

∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

Points in direction of maximum increase.

### Chain Rule

For composite functions, essential for backpropagation:

If y = f(u) and u = g(x), then:
dy/dx = (dy/du) * (du/dx)

### Computational Graphs and Automatic Differentiation

Modern frameworks (TensorFlow, PyTorch) use computational graphs to automatically compute gradients:

1. **Forward Pass:** Compute outputs, store intermediate values
2. **Backward Pass:** Apply chain rule to compute gradients

### Example: Simple Network

```
y = σ(Wx + b)
Loss = (y - target)²
```

Gradients:
```
∂Loss/∂y = 2(y - target)
∂y/∂z = σ'(z)  where z = Wx + b
∂z/∂W = x
∂Loss/∂W = (∂Loss/∂y) * (∂y/∂z) * (∂z/∂W)
         = 2(y - target) * σ'(z) * x
```

### Numerical Stability

Challenges in gradient computation:
- **Vanishing gradients:** Gradients become extremely small
- **Exploding gradients:** Gradients become extremely large

Solutions:
- Appropriate activation functions (ReLU)
- Gradient clipping
- Batch normalization
- Residual connections

### Matrix Calculus

For vector/matrix functions:

∂(Wx)/∂W = x (for scalar loss, more complex for vector outputs)

Jacobian matrices represent gradients for vector-valued functions.

## Summary

The mathematical foundations covered in this unit are essential for understanding and implementing deep learning algorithms. Tensor operations provide the computational framework, while gradient calculations enable learning through optimization. Mastery of these concepts allows for effective design, implementation, and debugging of neural networks.

Key takeaways:
- Neural networks are compositions of matrix operations and non-linear functions
- Tensors generalize scalars, vectors, and matrices to arbitrary dimensions
- Broadcasting enables efficient computation on arrays of different shapes
- Gradient calculation through backpropagation relies on the chain rule
- Modern frameworks automate gradient computation via computational graphs
