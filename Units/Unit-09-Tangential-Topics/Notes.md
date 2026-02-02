# Unit 09: Tangential Topics of Machine Learning and Neural Networks - Notes

## Introduction

This unit explores theoretical foundations and alternative learning paradigms that provide deeper insights into neural network behavior. Topics include information theory, biological learning principles, and specialized architectures that complement mainstream deep learning techniques.

## Information-Theoretic Machine Learning

### Overview

Information theory provides mathematical framework for understanding data, communication, and learning. Applying these concepts to machine learning reveals fundamental limits and principles.

### Relevance to Deep Learning

- Quantify information in data and representations
- Understand compression and generalization
- Analyze information flow through networks
- Theoretical bounds on learning

## Basic Concepts of Information Theory

### Entropy

Measures uncertainty or information content:

H(X) = -Σ p(x) log p(x)

**Properties:**
- H(X) ≥ 0
- Maximum when all outcomes equally likely
- Minimum (0) for deterministic variable

**Interpretation:**
Average number of bits needed to encode X

### Mutual Information

Measures shared information between variables:

I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

**Properties:**
- I(X; Y) ≥ 0
- I(X; Y) = 0 if X and Y independent
- Symmetric: I(X; Y) = I(Y; X)

**Interpretation:**
Reduction in uncertainty about X when Y is known

### Conditional Entropy

Uncertainty in X given Y:

H(X|Y) = -Σ p(x,y) log p(x|y)

### KL Divergence

Measures difference between distributions P and Q:

D_KL(P||Q) = Σ p(x) log(p(x)/q(x))

**Properties:**
- D_KL(P||Q) ≥ 0
- D_KL(P||Q) = 0 iff P = Q
- Not symmetric

**Applications:**
- Measure distribution difference
- Regularization in VAEs
- Optimization objective

## Information Processing in Analog Channels

### Channel Capacity

Maximum rate of reliable information transmission:

C = max_{p(x)} I(X; Y)

**Shannon's Theorem:**
For rate R < C, arbitrarily low error rate achievable

### Noise and Distortion

Channels introduce noise, limiting information transfer:
- Additive noise
- Multiplicative noise
- Distortion

### Deep Networks as Channels

Each layer can be viewed as information processing channel:
- Input carries information about task
- Layer processes and transforms information
- Output should retain task-relevant information

## Deep Neural Networks: Information Theoretical Perspective

### Information Plane

Plot mutual information:
- I(X; T): Information about input in layer T
- I(T; Y): Information about output in layer T

Visualizes information flow and compression through layers.

### Learning Dynamics

**Fitting Phase:**
Increase I(T; Y) (information about labels)

**Compression Phase:**
Decrease I(X; T) while maintaining I(T; Y)
Network discards task-irrelevant information

### Implications

- Good representations compress input while retaining task information
- Generalization tied to compression
- Different learning phases have different dynamics

## Information Bottleneck Methodology

### Principle

Find representation T of input X that:
- Maximizes information about output Y: I(T; Y)
- Minimizes information from input X: I(X; T)

**Objective:**

min_{p(t|x)} [I(X; T) - β I(T; Y)]

Where β controls trade-off.

### Application to Deep Learning

Neural networks naturally implement information bottleneck:
- Hidden layers compress input
- Retain task-relevant information
- Discard noise and irrelevant details

### Benefits

- **Generalization:** Compression prevents overfitting
- **Robustness:** Irrelevant variations ignored
- **Interpretability:** Compact representations easier to analyze

## Capacity Modelling Theorems

### VC Dimension

Measures capacity of hypothesis class:
- Maximum number of points that can be shattered (all possible labelings achieved)

**Implication:**
Relates model complexity to generalization error

### Rademacher Complexity

Measures richness of function class:
- Ability to fit random noise

**Application:**
Generalization bounds for deep networks

### Sample Complexity

Number of samples needed for learning:
- Depends on model capacity and desired accuracy
- Informally: More complex models need more data

## Characteristics of Deep Neural Network Layers

### Early Layers

- Extract low-level features (edges, textures)
- High information about input
- Less task-specific

### Middle Layers

- Combine low-level features into patterns
- Balance between input and output information
- Task-specific representations emerge

### Deep Layers

- Abstract, task-specific representations
- Lower information about raw input
- High information about target

## Phases in Double Optimization

### Empirical Risk Minimization

Optimize weights to minimize loss on training data:

min_θ Σ L(f(x_i; θ), y_i)

### Structural Risk Minimization

Balance training error and model complexity:

min_θ [Σ L(f(x_i; θ), y_i) + λ Complexity(θ)]

### Information Bottleneck View

First phase: Increase I(T; Y) (fit labels)
Second phase: Decrease I(X; T) (compress)

## Encoder and Decoder in DNN

### Encoder

Maps input to latent representation:
- Compresses information
- Retains task-relevant features

### Decoder

Maps latent representation to output:
- Reconstructs or predicts
- Uses compressed information

### Autoencoders

Encoder-decoder architecture for unsupervised learning:
- Learn compressed representations
- Reconstruction loss trains network

## Learning Theory

### PAC Learning

Probably Approximately Correct framework:
- With probability 1-δ, learned hypothesis has error ≤ ε
- Sample complexity bounds

### Bias-Variance Trade-off

Error = Bias² + Variance + Noise

**Bias:** Error from incorrect assumptions
**Variance:** Error from sensitivity to training data

### Regularization

Reduces variance, may increase bias slightly:
- Aims for optimal trade-off
- Improves generalization

## Stochastic Relation and Hidden Layers

### Stochastic Representations

Hidden layers produce stochastic outputs:
- Dropout introduces randomness
- Bayesian neural networks model weight uncertainty

### Benefits

- Exploration during training
- Uncertainty quantification
- Regularization effect

## Applying Information Gain

### Feature Selection

Choose features with high mutual information with target:
- I(Feature; Target) quantifies relevance

### Decision Trees

Split based on information gain:
- Reduction in entropy after split
- IG = H(Y) - H(Y|X)

### Neural Networks

Neurons or layers with low information gain can be pruned:
- Reduces complexity
- Improves efficiency

## Hebbian Learning

### Principle

"Cells that fire together, wire together"

Strengthen connections between simultaneously active neurons.

### Rule

Δw_ij = η x_i y_j

Where:
- w_ij: Weight from neuron i to j
- η: Learning rate
- x_i: Pre-synaptic activation
- y_j: Post-synaptic activation

### Biological Basis

Models synaptic plasticity observed in neuroscience.

## Implementation of Hebbian Learning in a Perceptron

### Algorithm

For each input pattern:
1. Compute output: y = w^T x
2. Update weights: w_new = w_old + η x y

### Example

Training on patterns:
- Weights strengthen for correlated inputs and outputs
- Simple, local learning rule

### Characteristics

- Unsupervised (no explicit target)
- Local (only uses pre- and post-synaptic activity)
- Simple and biologically plausible

## Limitations of Hebbian Learning

### Unbounded Growth

Weights can grow indefinitely:
- Unstable without normalization

**Solution:** Oja's rule, weight decay

### No Error Correction

Does not minimize specific loss:
- Cannot guarantee optimal performance
- No mechanism to correct mistakes

### Limited Applicability

Best for:
- Simple associations
- Unsupervised learning
- Biological modeling

Not suitable for:
- Complex supervised tasks
- Multi-layer networks (no clear error signal for hidden layers)

## Competitive Learning

### Concept

Neurons compete to respond to input:
- Winner takes all (or most)
- Only winner updates weights

Promotes specialization: Each neuron learns different pattern.

## What is Competition in Neural Networks?

### Mechanism

1. Compute similarity (or distance) between input and neuron weights
2. Select neuron with highest similarity (winner)
3. Update only winner's weights

### Outcome

- Neurons become selective to different input patterns
- Clustering effect
- Unsupervised feature learning

## Characteristics of Competitive Learning

- **Winner-Take-All:** Single winner per input
- **Unsupervised:** No external labels
- **Specialization:** Neurons differentiate
- **Self-Organization:** Structure emerges from data

## Criteria for Competitive Learning

### Requirements

1. Set of neurons compete
2. Limit on neuron strength (normalization)
3. Mechanism to select winner
4. Update rule favoring winner

### Lateral Inhibition

Winners inhibit nearby neurons:
- Enforces competition
- Prevents all neurons learning same pattern

## Architecture and Implementation

### Basic Structure

- Input layer: Presents patterns
- Competitive layer: Neurons compete

### Algorithm

```
For each input x:
    1. Compute distances: d_i = ||x - w_i||
    2. Find winner: j = argmin_i d_i
    3. Update winner: w_j = w_j + η(x - w_j)
    4. Normalize: w_j = w_j / ||w_j||
```

### Normalization

Ensure weights don't grow unbounded:
- Unit length constraint
- Prevents single neuron dominating

## Mathematical Representation

### Distance Metric

Euclidean distance:

d_i = ||x - w_i|| = √(Σ(x_k - w_{ik})²)

### Update Rule

Move winner's weights toward input:

Δw_j = η(x - w_j)

Only for winning neuron j.

## Competitive Learning Rule

### Standard Form

w_j(new) = w_j(old) + η(x - w_j(old))

### Vector Quantization Interpretation

Neurons represent cluster centers:
- Learn prototype vectors
- Input assigned to nearest prototype

### Self-Organizing Maps (SOM)

Extension of competitive learning:
- Neurons arranged in topological structure
- Neighbors of winner also update (less strongly)
- Preserves input space topology

## Hebbian Learning and Competitive Learning

### Relationship

Both biological inspiration, unsupervised:
- **Hebbian:** Correlation-based, cooperative
- **Competitive:** Winner-take-all, promotes diversity

### Combination

Can combine principles:
- Hebbian for local learning
- Competitive for global organization

## Boltzmann Learning

### Energy-Based Framework

Define energy function over network states:
- Low energy: Likely states
- High energy: Unlikely states

Learning adjusts weights to assign low energy to training examples.

## Boltzmann Machines

### Structure

- Visible units: Observed data
- Hidden units: Latent variables
- Connections: Between all units (fully connected variant)

### Energy Function

E(v, h) = -Σ_ij w_ij v_i h_j - Σ_i b_i v_i - Σ_j c_j h_j

Where:
- v: Visible units
- h: Hidden units
- w, b, c: Parameters

### Probability Distribution

P(v, h) = exp(-E(v, h)) / Z

Where Z is partition function (normalization).

### Learning

Maximize log-likelihood of data:
- Adjust weights to increase probability of observed data
- Uses Markov Chain Monte Carlo (MCMC) sampling
- Computationally expensive

## Energy-Based Models (EBMs)

### Concept

Define energy for configurations:
- Learn energy function that assigns low energy to data
- Generate samples from low-energy regions

### Applications

- Generative modeling
- Denoising
- Structured prediction

### Training

Contrastive learning:
- Push down energy for data samples
- Push up energy for non-data samples

## Restricted Boltzmann Machines (RBMs)

### Restriction

No connections within visible or hidden layers:
- Only between visible and hidden
- Bipartite graph structure

### Advantage

Conditional independence:
- P(h|v): Hidden units independent given visible
- P(v|h): Visible units independent given hidden

Enables efficient inference.

## Restricted Boltzmann Machines - Working

### Energy Function

E(v, h) = -v^T W h - b^T v - c^T h

### Conditional Distributions

P(h_j = 1 | v) = σ(c_j + Σ_i w_ij v_i)
P(v_i = 1 | h) = σ(b_i + Σ_j w_ij h_j)

Where σ is sigmoid function.

### Training: Contrastive Divergence

Approximate gradient with Gibbs sampling:
1. Start with training sample v_0
2. Compute P(h|v_0), sample h_0
3. Compute P(v|h_0), sample v_1
4. Compute P(h|v_1), sample h_1
5. Update weights:

Δw_ij = η (v_0 h_0^T - v_1 h_1^T)

### Applications

- Pre-training deep networks (historical importance)
- Collaborative filtering
- Feature learning
- Dimensionality reduction

## Radial-Basis Function Networks (RBFNs)

### Concept

Network with RBF activation in hidden layer:
- Neurons respond to local regions of input space
- Localized, non-linear transformations

## RBF Network Architecture

### Structure

**Input Layer:** Receives input vector

**Hidden Layer:**
- RBF neurons with centers c_i and widths σ_i
- Activation: φ_i(x) = exp(-||x - c_i||² / (2σ_i²))

**Output Layer:**
- Linear combination of hidden activations
- y = Σ w_i φ_i(x)

### Parameters

- Centers c_i: Locations in input space
- Widths σ_i: Spread of RBF
- Weights w_i: Output layer weights

## RBF Neuron Activation Function

### Gaussian RBF

φ(x) = exp(-||x - c||² / (2σ²))

**Properties:**
- Maximum at center c
- Decreases with distance
- Localized response

### Other RBFs

- Multiquadric: √(||x - c||² + σ²)
- Inverse multiquadric: 1 / √(||x - c||² + σ²)
- Thin plate spline: ||x - c||² log(||x - c||)

## RBFN as a Neural Network

### Universal Approximation

RBFNs can approximate any continuous function with sufficient hidden units.

### Training

**Two-Stage:**
1. Set centers and widths (unsupervised):
   - K-means clustering
   - Random selection
   - Evenly spaced
2. Train output weights (supervised):
   - Linear regression (closed-form solution)

**Joint Optimization:**
- Gradient descent on all parameters
- More flexible but harder to train

### Comparison to MLP

| Aspect | RBFN | MLP |
|--------|------|-----|
| Activation | Localized (RBF) | Global (sigmoid, ReLU) |
| Training | Often two-stage | Backpropagation |
| Hidden layers | Typically one | Multiple |
| Interpretability | Centers are prototypes | Less interpretable |

## Advantages of Using RBFN than ML

### Advantages

**1. Faster Training:**
Two-stage approach often faster than iterative backpropagation

**2. Local Approximation:**
Changes in one region don't affect distant regions

**3. Interpretability:**
Centers represent prototypes or landmarks

**4. Simple Output Training:**
Linear output layer (closed-form solution)

**5. Universal Approximation:**
Guaranteed with sufficient neurons

### Disadvantages

**1. Curse of Dimensionality:**
Number of centers grows exponentially with input dimensions

**2. Center Selection:**
Performance sensitive to center placement

**3. Shallow Architecture:**
Typically one hidden layer (less hierarchical learning)

**4. Scalability:**
May require many RBF units for complex functions

### When to Use

- Small to moderate input dimensions
- Function approximation tasks
- When fast training preferred
- Interpretability important

## Summary

This unit explored theoretical and alternative perspectives on neural networks:

- **Information Theory:** Provides framework for understanding learning, compression, and generalization
- **Information Bottleneck:** Explains representation learning as balancing informativeness and compression
- **Hebbian Learning:** Biologically inspired, correlation-based learning with limitations
- **Competitive Learning:** Unsupervised specialization through competition, basis for clustering
- **Boltzmann Machines:** Energy-based generative models, foundational for modern approaches
- **RBFNs:** Localized activation functions for function approximation with interpretable centers

These concepts complement mainstream deep learning, offering theoretical insights and alternative architectures for specialized applications. Understanding multiple perspectives enriches one's ability to design, analyze, and improve neural network systems.
