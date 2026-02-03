# Handwritten Digit Classification Using CNN

##  Project Overview

This project implements a **Neural Network model** to classify handwritten digits (0–9) using the **MNIST dataset**. It demonstrates core deep learning concepts such as data preprocessing, model construction, training, and evaluation using **TensorFlow** and **Keras**.

The model takes a **28×28 grayscale image** of a handwritten digit as input and predicts the correct digit with high accuracy.

---

##  Dataset

The project uses the **MNIST (Modified National Institute of Standards and Technology) dataset**:

* **Training Images:** 60,000
* **Test Images:** 10,000
* **Image Size:** 28 × 28 pixels (grayscale)
* **Classes:** Digits from 0 to 9

---

##  Workflow

### 1. Data Loading

* Load the MNIST dataset directly using `keras.datasets`

### 2. Preprocessing

* **Normalization:** Scale pixel values from 0–255 to 0–1
* **Flattening:** Convert 2D images (28×28) into 1D vectors (784 features)

---

##  Model Architecture

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
```

* **Input Layer:** Flattens 28×28 images
* **Hidden Layers:** Dense layers with ReLU activation
* **Output Layer:** 10 neurons for digit classification

---

##  Training Details

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metric:** Accuracy

---

## Evaluation

* Model is tested on unseen test data
* Performance measured using accuracy
* Confusion matrix is plotted to analyze predictions

---

##  Results

The model achieves **high accuracy** on the MNIST test dataset, showing effective handwritten digit classification.

---

## Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

