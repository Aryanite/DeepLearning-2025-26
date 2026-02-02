# Unit 06: Recurrent Neural Networks (RNN) - Notes

## Introduction

Recurrent Neural Networks (RNNs) are neural network architectures designed to process sequential data by maintaining internal memory. Unlike feedforward networks, RNNs have connections that form cycles, allowing information to persist and enabling them to learn temporal dependencies.

## Recurrent Neural Network (RNN)

### Definition

RNNs process sequences by iterating through sequence elements and maintaining a hidden state that captures information about previous elements.

### Basic Structure

At each time step t:
- Input: x_t
- Hidden state: h_t
- Output: y_t

The hidden state is updated based on current input and previous hidden state:

h_t = f(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y

Where:
- W_hh: Hidden-to-hidden weights
- W_xh: Input-to-hidden weights
- W_hy: Hidden-to-output weights
- f: Activation function (typically tanh or ReLU)

## Why Recurrent Networks?

### Sequential Data Characteristics

Many real-world problems involve sequences:
- Time series (stock prices, weather)
- Natural language (text, speech)
- Video (sequences of frames)
- DNA sequences
- Music

### Temporal Dependencies

Sequences exhibit dependencies across time:
- Current output may depend on inputs from many time steps ago
- Context matters for interpretation

### Advantages of RNNs

- **Variable-length inputs:** Can process sequences of any length
- **Parameter sharing:** Same weights applied at each time step
- **Memory:** Hidden state acts as memory of past inputs
- **Temporal modeling:** Explicitly designed for sequential patterns

## RNN Explained

### Unrolled Representation

An RNN can be "unrolled" across time, revealing its structure:

```
x_1 → [RNN] → y_1
      ↓
x_2 → [RNN] → y_2
      ↓
x_3 → [RNN] → y_3
```

At each step, the same weights are used, but hidden state carries information forward.

### Types of RNN Architectures

**One-to-One:**
Standard neural network (not truly recurrent)

**One-to-Many:**
Single input, sequence output (e.g., image captioning)

**Many-to-One:**
Sequence input, single output (e.g., sentiment classification)

**Many-to-Many (synchronized):**
Sequence input and output, same length (e.g., video frame labeling)

**Many-to-Many (encoder-decoder):**
Sequence input and output, different lengths (e.g., machine translation)

## Deep RNNs

### Concept

Stack multiple RNN layers:
- First layer processes input sequence
- Subsequent layers process outputs of previous layer
- Allows hierarchical temporal feature learning

### Architecture

```
Input → RNN Layer 1 → RNN Layer 2 → ... → RNN Layer N → Output
```

Each layer has its own hidden state.

### Benefits

- Learn more complex patterns
- Hierarchical representations
- Often better performance on complex tasks

## Recursive Neural Networks

### Distinction from Recurrent Networks

- **Recurrent:** Process sequences linearly
- **Recursive:** Process hierarchical structures (trees)

### Applications

- Natural language parsing (sentence structure trees)
- Scene understanding (object hierarchies)

## Step Function and Tanh Function

### Step Function

```
f(x) = 1 if x >= 0
       0 if x < 0
```

- Not differentiable at 0
- Rarely used in modern neural networks
- Binary output

### Tanh Function

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- Output range: (-1, 1)
- Zero-centered
- Smooth gradient
- Commonly used in RNN hidden states
- Suffers from vanishing gradient for extreme values

## RNN in Memory

### Hidden State as Memory

The hidden state h_t encodes information from all previous time steps, acting as the network's memory.

### Limitations

**Vanishing Gradient Problem:**
- Gradients diminish exponentially with sequence length
- Difficult to learn long-term dependencies
- Information from distant past gets lost

**Solution:** LSTM and GRU architectures

## Long Short Term Memory (LSTM)

### Motivation

LSTMs address the vanishing gradient problem through gating mechanisms that regulate information flow.

### Key Innovation

Separate cell state from hidden state:
- **Cell state (C_t):** Long-term memory
- **Hidden state (h_t):** Short-term memory/output

## Working Components of LSTMs

### Four Main Components

**1. Forget Gate (f_t):**
Decides what information to discard from cell state

f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

**2. Input Gate (i_t):**
Decides what new information to add to cell state

i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

**3. Cell State Update:**
Candidate values for cell state

C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

Update cell state:

C_t = f_t * C_{t-1} + i_t * C̃_t

**4. Output Gate (o_t):**
Decides what information to output

o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)

Where σ is sigmoid function.

## Core Idea Behind LSTMs

### Controlled Information Flow

Gates use sigmoid activations (output 0 to 1):
- 0: Block all information
- 1: Pass all information
- Intermediate: Partial passage

### Learning What to Remember and Forget

Through training, LSTM learns:
- Which past information is relevant (forget gate)
- Which new information is important (input gate)
- What to output at each step (output gate)

### Gradient Flow

Cell state provides a "highway" for gradients:
- Additive updates preserve gradients
- Mitigates vanishing gradient problem
- Enables learning long-term dependencies

## LSTM: A Simple Walk Through

### Step-by-Step Process

**Given:**
- Previous hidden state: h_{t-1}
- Previous cell state: C_{t-1}
- Current input: x_t

**1. Forget Gate:**
Determine what to forget from C_{t-1}

**2. Input Gate:**
Determine what new information to add

**3. Update Cell State:**
Remove forgotten information, add new information

**4. Output Gate:**
Determine current hidden state based on cell state

**5. Produce Output:**
Use hidden state for prediction

## Gated Recurrent Unit (GRU)

### Motivation

GRU simplifies LSTM while maintaining performance:
- Fewer parameters
- Faster training
- Often comparable results

### Structure

GRU has two gates instead of three:

**1. Reset Gate (r_t):**
Controls how much past information to forget

r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

**2. Update Gate (z_t):**
Controls how much past information to keep

z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

**3. Candidate Hidden State:**

h̃_t = tanh(W · [r_t * h_{t-1}, x_t] + b)

**4. Final Hidden State:**

h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

## GRU Design Steps

### Simplifications from LSTM

- No separate cell state
- Fewer gates (2 vs 3)
- Single hidden state serves dual purpose

### Update Mechanism

Update gate determines balance between:
- Retaining old hidden state
- Accepting new candidate state

## Fully Gated vs Minimal Gated Architecture

### Fully Gated (Standard GRU)

Uses both reset and update gates as described above.

### Minimal Gated (MGU)

Further simplification:
- Only one gate
- Even fewer parameters
- May sacrifice some performance

### Trade-offs

- Fully gated: Better capacity, more parameters
- Minimal gated: Faster, simpler, potentially underfitting

## Working of RNNs

### Forward Pass

Iterate through sequence:
1. Start with initial hidden state (often zeros)
2. For each time step:
   - Compute hidden state based on input and previous hidden state
   - Compute output
3. Use outputs for loss computation

### Training Challenges

- **Vanishing gradients:** Gradients decay with backpropagation through time
- **Exploding gradients:** Gradients grow exponentially
- **Long-term dependencies:** Difficulty capturing relationships across many time steps

## Backpropagation Through Time (BPTT)

### Concept

Unroll RNN across time and apply backpropagation to the unrolled network.

### Algorithm

1. **Forward pass:** Compute outputs and hidden states for all time steps
2. **Backward pass:** Compute gradients by propagating error backward through time
3. **Update weights:** Use accumulated gradients

### Gradient Computation

For loss at time T:

∂L/∂W = Σ_{t=1}^{T} ∂L_t/∂W

Gradients flow backward through recurrent connections.

### Truncated BPTT

Limit backpropagation to k steps:
- Reduces computational cost
- Mitigates gradient issues
- May miss very long-term dependencies

## Backpropagation Through Computational Graphs

### Computational Graph View

Represent RNN as directed acyclic graph (when unrolled):
- Nodes: Operations, variables
- Edges: Dependencies

### Automatic Differentiation

Modern frameworks (TensorFlow, PyTorch) automatically compute gradients:
- Forward mode: Compute outputs
- Reverse mode: Compute gradients via chain rule

## Complex Recurrent Neural Networks

### Architectural Variations

- **Bidirectional RNNs:** Process sequence forward and backward
- **Attention mechanisms:** Selectively focus on relevant parts
- **Encoder-decoder:** Sequence-to-sequence models
- **Multi-task RNNs:** Shared representations for multiple tasks

## Over-fitting and Under-fitting

### Over-fitting in RNNs

**Signs:**
- Low training error, high validation error
- Model memorizes training sequences
- Poor generalization

**Causes:**
- Too many parameters relative to data
- Insufficient regularization
- Training too long

### Under-fitting in RNNs

**Signs:**
- High training and validation error
- Model too simple for task

**Causes:**
- Insufficient capacity
- Too few layers/units
- Under-training

## Detect and Avoid Overfitting

### Detection

- Monitor validation loss during training
- Plot training vs validation curves
- Gap between curves indicates overfitting

### Prevention Strategies

**1. Regularization:**
- Dropout (applied to inputs, outputs, recurrent connections)
- L2 weight decay

**2. Early Stopping:**
Stop training when validation loss stops improving

**3. Data Augmentation:**
Increase effective dataset size

**4. Reduce Model Complexity:**
Fewer layers, smaller hidden size

**5. More Training Data:**
Collect additional examples

## Prevention of Overfitting: Approach on Model and Data

### Model-based Approaches

- **Dropout:** Randomly deactivate units during training
- **Recurrent dropout:** Apply dropout to recurrent connections
- **Weight constraints:** Limit weight magnitudes
- **Batch normalization:** Normalize layer inputs

### Data-based Approaches

- **Data augmentation:** Add noise, perturbations
- **Cross-validation:** Ensure robust performance estimate
- **Balanced datasets:** Avoid class imbalance

## Multi-layered RNNs

### Stacked Architecture

Multiple RNN layers stacked vertically:

```
Input → RNN1 → RNN2 → ... → RNNN → Output
```

### Benefits

- Learn hierarchical temporal features
- Lower layers: Simple patterns
- Upper layers: Complex, abstract patterns
- Generally improved performance

### Implementation

```python
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(num_classes, activation='softmax')
])
```

## Stacked LSTM

### Architecture

Multiple LSTM layers:
- Each layer processes outputs of previous layer
- Each maintains own hidden and cell states
- Increased model capacity

### Best Practices

- Use `return_sequences=True` for all but last LSTM layer
- Start with 2-3 layers, add more if needed
- Monitor for overfitting

## Multi-directional RNNs

### Bidirectional RNNs

Process sequence in both directions:
- **Forward RNN:** Left to right
- **Backward RNN:** Right to left
- Concatenate or combine outputs

### Formula

h_t = [h⃗_t; h⃖_t]

Where h⃗_t is forward hidden state, h⃖_t is backward hidden state.

### Applications

Effective when full sequence is available:
- Sentence classification
- Named entity recognition
- Speech recognition

Not suitable for real-time prediction (requires future context).

## Difference Between LSTM and BI-LSTM

### LSTM

- Processes sequence in one direction
- Access to past context only
- Suitable for online/real-time tasks

### Bidirectional LSTM (BI-LSTM)

- Processes sequence in both directions
- Access to past and future context
- Better for offline tasks
- Approximately double parameters and computation

## One-dimensional Sequence Processing

### 1D Convolutions for Sequences

Apply convolution along time dimension:
- Faster than RNNs
- Parallel computation
- Limited temporal receptive field

### Hybrid Approaches

Combine CNNs and RNNs:
- CNN for local feature extraction
- RNN for global temporal modeling

## CNN and RNN Comparison

### Convolutional Neural Networks

**Strengths:**
- Parallel computation
- Translation invariant
- Efficient for images
- Fixed receptive field

**Weaknesses:**
- Limited temporal modeling
- Fixed input size (can be addressed)

### Recurrent Neural Networks

**Strengths:**
- Variable-length sequences
- Temporal dependencies
- Maintain state/memory
- Flexible receptive field

**Weaknesses:**
- Sequential computation (slow)
- Vanishing/exploding gradients
- Difficult to parallelize

### Choosing Between Them

- **Images:** CNNs
- **Long sequences, temporal dependencies:** RNNs (LSTM/GRU)
- **Short sequences, local patterns:** 1D CNNs
- **Hybrid:** Use both (CNN for features, RNN for temporal modeling)

## Summary

Recurrent Neural Networks enable processing of sequential data through recurrent connections and hidden states that maintain memory. Key developments include:

- Vanilla RNNs process sequences but suffer from vanishing gradients
- LSTMs use gating mechanisms to learn long-term dependencies
- GRUs simplify LSTMs while maintaining effectiveness
- Backpropagation through time trains RNNs by unrolling across time
- Stacked and bidirectional architectures increase model capacity
- Regularization techniques prevent overfitting in RNNs

Understanding RNNs and their variants is essential for tackling sequence-based tasks in natural language processing, time series analysis, and many other domains. Modern applications often combine RNNs with attention mechanisms and transformers for even better performance.
