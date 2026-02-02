# Unit 07: Generative Deep Learning - Notes

## Introduction

Generative deep learning focuses on models that can create new data samples resembling training data. Unlike discriminative models that learn decision boundaries, generative models learn the underlying data distribution to generate novel, realistic samples.

## Generative Deep Learning

### Definition

Generative models learn probability distribution P(X) of data, enabling:
- Sampling new data points
- Density estimation
- Understanding data structure

### Applications

- Text generation and completion
- Image synthesis and editing
- Music composition
- Drug discovery
- Data augmentation
- Anomaly detection

### Types of Generative Models

1. **Autoregressive Models:** Generate data sequentially (e.g., LSTMs for text)
2. **Variational Autoencoders (VAEs):** Learn latent representations
3. **Generative Adversarial Networks (GANs):** Adversarial training approach
4. **Normalizing Flows:** Invertible transformations
5. **Diffusion Models:** Iterative denoising process

## Using LSTMs to Synthesize Text

### Concept

LSTMs can model text as sequences, predicting next character/word given context:

P(x_t | x_1, x_2, ..., x_{t-1})

### Architecture

```
Input: Character/word sequence
LSTM layers: Learn sequential patterns
Output: Probability distribution over next character/word
```

### Training

1. Feed text corpus to LSTM
2. Train to predict next element
3. Use cross-entropy loss
4. Model learns language patterns, grammar, style

## Text Synthetization Procedures

### Character-Level Generation

**Training:**
- Input: Sequence of characters
- Output: Next character prediction
- Vocabulary: All unique characters

**Generation:**
1. Provide seed text
2. Predict next character probabilistically
3. Append to sequence
4. Repeat

### Word-Level Generation

**Training:**
- Input: Sequence of words
- Output: Next word prediction
- Vocabulary: All unique words (or top N)

**Advantages:**
- More coherent output
- Faster generation

**Disadvantages:**
- Larger vocabulary
- Cannot generate new words

### Sampling Strategies

**Greedy Sampling:**
Always choose most probable next element
- Deterministic
- Can be repetitive

**Temperature Sampling:**
Adjust probability distribution:

P'(x) = P(x)^(1/T) / Σ P(x_i)^(1/T)

- Low T: Conservative, likely choices
- High T: Diverse, creative choices
- T = 1: Original distribution

**Top-k Sampling:**
Sample from k most likely choices

**Nucleus (Top-p) Sampling:**
Sample from smallest set with cumulative probability ≥ p

### Implementation Example

```python
model = keras.Sequential([
    layers.LSTM(128, input_shape=(maxlen, vocab_size), return_sequences=True),
    layers.LSTM(128),
    layers.Dense(vocab_size, activation='softmax')
])

# Generate text
def generate_text(seed, length, temperature=1.0):
    generated = seed
    for _ in range(length):
        x = encode(generated[-maxlen:])
        preds = model.predict(x)[0]
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_char = np.random.choice(vocab_size, p=preds)
        generated += decode(next_char)
    return generated
```

## Neural Style Transfer (NST)

### Concept

Combine content of one image with artistic style of another:
- Content image: Photograph
- Style image: Artwork
- Output: Content in artistic style

### Applications

- Artistic image creation
- Video stylization
- Real-time filters
- Creative tools

## NST Working Principle

### Key Idea

Use pre-trained CNN (e.g., VGG19) as feature extractor:
- **Content representation:** High-level features from deep layers
- **Style representation:** Correlations between features across layers

### Objective Function

Minimize combined loss:

L_total = α * L_content + β * L_style

Where:
- α, β: Weighting factors
- L_content: Content loss
- L_style: Style loss

## Content and Style Management in NST

### Content Loss

Preserve content from content image:

L_content = Σ(F^l - P^l)²

Where:
- F^l: Features of generated image at layer l
- P^l: Features of content image at layer l

Use features from deep layers (capture high-level content).

### Style Loss

Capture style through Gram matrices:

G^l_{ij} = Σ_k F^l_{ik} * F^l_{jk}

Gram matrix captures correlations between feature maps.

Style loss:

L_style = Σ_l w_l * Σ_{ij} (G^l_{ij} - A^l_{ij})²

Where:
- G^l: Gram matrix of generated image at layer l
- A^l: Gram matrix of style image at layer l
- w_l: Weight for layer l

Use features from multiple layers (capture multi-scale style).

## NST Implementation

### Steps

1. **Load Pre-trained Model:**
   VGG19 trained on ImageNet

2. **Define Content and Style Layers:**
   ```python
   content_layers = ['block5_conv2']
   style_layers = ['block1_conv1', 'block2_conv1', 
                   'block3_conv1', 'block4_conv1', 'block5_conv1']
   ```

3. **Extract Features:**
   Pass content, style, and generated images through model

4. **Compute Losses:**
   Calculate content and style losses

5. **Optimization:**
   Initialize generated image (e.g., with content image)
   Use gradient descent to minimize total loss
   Update generated image pixels

### Pseudo-code

```python
# Initialize generated image
generated = content_image.copy()

# Optimization loop
for iteration in range(num_iterations):
    with tf.GradientTape() as tape:
        content_loss = compute_content_loss(generated, content_features)
        style_loss = compute_style_loss(generated, style_features)
        total_loss = content_weight * content_loss + style_weight * style_loss
    
    gradients = tape.gradient(total_loss, generated)
    optimizer.apply_gradients([(gradients, generated)])
```

## Image Synthesis with Variational Autoencoders

### Need for Image Synthesis

- **Data Augmentation:** Generate training data
- **Creative Applications:** Art, design
- **Compression:** Learn compact representations
- **Understanding:** Model data distribution
- **Imputation:** Fill missing data

## Working Models

### Autoencoder Basics

**Encoder:** Maps input to latent representation
**Decoder:** Reconstructs input from latent representation

Traditional autoencoders learn deterministic mappings, limiting generation capability.

## Variational Auto Encoders (VAE)

### Key Innovation

Encoder produces probability distribution over latent space (typically Gaussian):

q(z|x) ~ N(μ(x), σ²(x))

Where encoder outputs mean μ and variance σ².

### Architecture

**Encoder:**
- Input: Image
- Output: μ and log(σ²) for latent distribution

**Sampling:**
- Sample z from N(μ, σ²) using reparameterization trick:
  z = μ + σ * ε, where ε ~ N(0, 1)

**Decoder:**
- Input: Latent vector z
- Output: Reconstructed image

### Loss Function

VAE loss combines reconstruction and regularization:

L = L_reconstruction + β * L_KL

**Reconstruction Loss:**
Pixel-wise difference (MSE, binary cross-entropy)

**KL Divergence:**
Regularizes latent distribution to match prior N(0, I):

L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

Encourages smooth, continuous latent space.

### Reparameterization Trick

Enables backpropagation through stochastic sampling:
Instead of sampling z ~ N(μ, σ²), sample ε ~ N(0, 1) and compute z = μ + σ * ε.

### Training

1. Encode input to μ and σ
2. Sample latent vector z
3. Decode z to reconstruct input
4. Compute reconstruction and KL losses
5. Backpropagate and update weights

### Generation

Sample z from prior N(0, I) and decode to create new images.

## Latent Space

### Concept

Latent space is lower-dimensional representation where data is encoded.

### Properties in VAE

- **Continuous:** Smooth interpolations possible
- **Structured:** Similar data points cluster together
- **Disentangled (ideally):** Each dimension captures independent factor of variation

### Applications

**Interpolation:**
Smoothly transition between images by interpolating in latent space

**Arithmetic:**
Perform operations in latent space (e.g., z_smiling = z_face + z_smile)

**Exploration:**
Sample from latent space to generate diverse samples

## Generative Adversarial Networks (GANs)

### Concept

Two neural networks compete:
- **Generator (G):** Creates fake samples
- **Discriminator (D):** Distinguishes real from fake

Through adversarial training, generator learns to produce increasingly realistic samples.

## Generative and Discriminative Algorithms

### Discriminative Models

Learn P(Y|X): Probability of label given input
- Focus on decision boundary
- Examples: Logistic regression, standard classifiers

### Generative Models

Learn P(X) or P(X|Y): Probability distribution of data
- Model data generation process
- Can sample new data
- Examples: VAEs, GANs, autoregressive models

## Applications Using GAN

- **Image Generation:** Photorealistic faces, objects
- **Image-to-Image Translation:** Style transfer, colorization, super-resolution
- **Data Augmentation:** Generate training samples
- **Anomaly Detection:** Identify outliers
- **Drug Discovery:** Generate molecular structures
- **Art and Design:** Creative content generation

## GAN Working Principle

### Minimax Game

Generator and discriminator play two-player game:

min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

**Discriminator objective:** Maximize ability to distinguish real from fake
**Generator objective:** Minimize discriminator's ability (equivalently, maximize probability discriminator is fooled)

### Training Dynamics

1. **Discriminator Training:**
   - Real samples: Maximize D(x) toward 1
   - Fake samples: Maximize D(G(z)) toward 0

2. **Generator Training:**
   - Generate fake samples
   - Maximize D(G(z)) toward 1 (fool discriminator)

3. **Equilibrium:**
   - Ideally, G produces perfect samples
   - D cannot distinguish (outputs 0.5)

## Generator and Discriminator

### Generator

**Input:** Random noise vector z (latent code)
**Output:** Generated sample G(z)

**Architecture (example for images):**
- Dense layer to expand noise
- Reshape to small feature maps
- Transposed convolutions (upsampling)
- Output layer with appropriate activation (tanh, sigmoid)

### Discriminator

**Input:** Sample x (real or generated)
**Output:** Probability that x is real

**Architecture (example for images):**
- Convolutional layers (downsampling)
- Flatten
- Dense layers
- Output: Single unit with sigmoid activation

## Training GAN

### Challenges

- **Mode Collapse:** Generator produces limited diversity
- **Vanishing Gradients:** Discriminator too strong, generator receives no gradient
- **Instability:** Training can be unstable, oscillate

### Best Practices

1. **Normalize Inputs:** Scale to [-1, 1] or [0, 1]
2. **Batch Normalization:** Stabilize training (in generator, sometimes in discriminator)
3. **LeakyReLU:** Use in discriminator to avoid sparse gradients
4. **Separate Learning Rates:** Often different for G and D
5. **Label Smoothing:** Use soft labels (0.9 instead of 1) for discriminator
6. **Gradient Penalties:** Regularize discriminator (e.g., Wasserstein GAN-GP)

### Algorithm

```
For each training iteration:
    1. Train Discriminator:
       - Sample minibatch of real data
       - Sample minibatch of noise, generate fake data
       - Compute discriminator loss on real and fake
       - Update discriminator weights
    
    2. Train Generator:
       - Sample minibatch of noise
       - Generate fake data
       - Compute generator loss (discriminator's output on fake)
       - Update generator weights
```

Often train discriminator multiple times per generator update.

## Implementing GAN: 1st Generation

### Simple GAN Implementation

```python
# Generator
generator = keras.Sequential([
    layers.Dense(256, input_dim=latent_dim),
    layers.LeakyReLU(0.2),
    layers.BatchNormalization(),
    layers.Dense(512),
    layers.LeakyReLU(0.2),
    layers.BatchNormalization(),
    layers.Dense(1024),
    layers.LeakyReLU(0.2),
    layers.BatchNormalization(),
    layers.Dense(image_size, activation='tanh')
])

# Discriminator
discriminator = keras.Sequential([
    layers.Dense(1024, input_dim=image_size),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Dense(512),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Dense(256),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Training loop
for epoch in range(epochs):
    # Train discriminator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    real_images = sample_real_images(batch_size)
    
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
    
    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
```

### Variants

- **DCGAN:** Uses convolutional layers, architectural guidelines
- **WGAN:** Wasserstein distance, improved training stability
- **StyleGAN:** High-quality image generation with style control
- **CycleGAN:** Unpaired image-to-image translation
- **Conditional GAN:** Generate samples conditioned on labels

## Summary

Generative deep learning enables creation of new data samples across domains:

- **LSTMs** generate text by modeling sequential dependencies
- **Neural Style Transfer** combines content and style using CNN features
- **VAEs** learn structured latent spaces for image generation
- **GANs** use adversarial training to produce highly realistic samples

Each approach has strengths:
- LSTMs: Effective for sequences
- NST: Combines existing images creatively
- VAEs: Smooth latent space, stable training
- GANs: High-quality, realistic outputs

Understanding generative models opens possibilities for creative AI applications, data augmentation, and insights into learned representations.
