# LeNet-5: Original 1998 Implementation in PyTorch

This repository contains a almost faithful reconstruction of the **LeNet-5** architecture as described in the seminal paper: *"Gradient-Based Learning Applied to Document Recognition"* (LeCun et al., 1998).

## 🚀 Unique Features
Unlike "modern" LeNet variants, this implementation includes the original paper's specific details:
* **Scaled Tanh Activation:** Uses $1.7159 \cdot \tanh(\frac{2}{3}x)$ to maintain variance and prevent saturation.
* **Custom C3 Connectivity:** Implements the specific sparse connection table between S2 and C3 feature maps.
* **Linear Subsampling (S2/S4):** Uses learnable coefficients and biases rather than simple Max-Pooling.
* **RBF Output Layer:** Uses Euclidean Distance (penalties) against fixed 12x7 digit prototypes instead of standard Softmax.
* **Discriminative Loss:** Implements the original log-sum-exp penalty function to push incorrect class energies away.

## 🛠️ Architecture Summary
1.  **C1:** Convolutional Layer (6 filters, 5x5)
2.  **S2:** Subsampling Layer (Average Pooling + Learnable Gain/Bias)
3.  **C3:** Sparse Convolutional Layer (16 filters, custom mapping)
4.  **S4:** Subsampling Layer
5.  **C5:** Convolutional Layer (120 filters, 5x5)
6.  **F6:** Fully Connected Layer (84 units)
7.  **Output:** RBF Layer (10 units, fixed prototypes)

## 📊 Getting Started
### Prerequisites
```bash
pip install torch torchvision numpy prettytable matplotlib scikit-learn
```

### Running the Model
```bash
    python src/main.py
```
## 🧠 Key Insights
The model uses a **penalty-based approach**. Predictions are made by selecting the class with the **minimum energy (distance)** in the output vector:

$$
\text{prediction} = \arg\min(\text{output})
$$