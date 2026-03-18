# 🚦 High-Def Traffic Light AI Detector

A Deep Learning computer vision application that classifies traffic light signals into 7 distinct categories using a custom Convolutional Neural Network (CNN). **Now featuring an interactive web UI powered by Gradio!**

## 🌟 Key Features

* **Custom CNN Architecture:** Built with TensorFlow & Keras for robust feature extraction.

* **7-Class Detection:** Accurately distinguishes between specific states like `stopLeft`, `goForward`, and `warning`.

* **Interactive Web App:** Includes a fully functional UI where users can upload their own images for real-time AI predictions.

* **Shareable Links:** Generates a public URL instantly via Gradio to share the detector with others.

## 📋 Table of Contents

* [Dataset](#-dataset)

* [Model Architecture](#-model-architecture)

* [The Web Interface](#-the-web-interface)

* [Getting Started](#-getting-started)

* [Results & Evaluation](#-results--evaluation)

## 📊 Dataset

The model is trained on the **Cropped LISA Traffic Light Dataset**.

* **Input Shape:** Images are dynamically resized to `256 x 256` pixels.

* **Batch Size:** `32`

* **Classes:** The dataset spans **7 distinct states**:

  1. 🟢 `go`

  2. ⬆️ `goForward`

  3. ⬅️ `goLeft`

  4. 🔴 `stop`

  5. 🛑 `stopLeft`

  6. 🟡 `warning`

  7. 🟨 `warningLeft`

## 🧠 Model Architecture

The deep learning model is built using the Keras `Sequential` API:

1. **Input Layer:** Standardized `256x256x3` RGB images.

2. **Convolutional & Pooling Layers (`Conv2D` + `MaxPooling2D`):** Extracts spatial hierarchies of features from the raw images.

3. **Regularization (`Dropout`):** Helps the model generalize better to unseen traffic lights.

4. **Classification Head (`Flatten` + `Dense`):** Outputs a probability distribution across the 7 possible traffic light states.

## 🌐 The Web Interface

This project utilizes **Gradio** to transform the trained model into an accessible web application.

Instead of writing code to test a new image, you can simply drag and drop a photo into the browser. The app handles the `256x256` resizing and normalization automatically, then displays the Top 3 predicted classes alongside their confidence percentages.

## 🚀 Getting Started

Follow these steps to run the training and launch the web app locally:

### Prerequisites

Make sure you have Python 3.x installed.

### Installation

1. **Clone the repository**

   ```
   git clone [https://github.com/Aryanite/DeepLearning-2025-26/tree/main/Projects/Traffic Light Detection.git](https://github.com/Aryanite/DeepLearning-2025-26.git)]
   cd Projects/Traffic Light Detection
   ```

2. **Install the required dependencies**

   ```
   pip install tensorflow numpy matplotlib seaborn scikit-learn gradio jupyter
   ```

3. **Launch the Notebook**

   ```
   jupyter notebook trafic-detection.ipynb
   ```

4. **Run the Interface**
   Execute all cells in the notebook. The final cell will launch the Gradio app:

   ```
   demo.launch(share=True)
   ```

   *A local URL (e.g., `http://127.0.0.1:7860`) and a public shareable URL will be printed in your console!*

## 📈 Results & Evaluation

Before deploying to the Gradio app, the model's performance is rigorously evaluated:

* **Confusion Matrix:** Rendered using `seaborn` to visualize precision and recall across all 7 classes, ensuring the model doesn't critically confuse a `stop` signal with a `go` signal.
