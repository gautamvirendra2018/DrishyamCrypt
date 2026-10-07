# DrishyamCrypt

**Drishyam** means **Vision** in Sanskrit language

**A Visualization Framework for Recurrence Plot-Based Neural Differential Cryptanalysis**

[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-green)](https://zenodo.org/records/18223783)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DrishyamCrypt** transforms 1D ciphertext-difference sequences into 2D recurrence-plot texture images, enabling superior CNN-based differential cryptanalysis distinguishers on reduced-round Ascon and Speck32/64.

## 🚀 Key Contributions

### 1. Recurrence Plot Representation

Transforms 1D ciphertext differences $\rightarrow$ 2D texture images exposing:

* **Diagonals:** Diffusion strength (long=weak, short=strong)
* **Laminar bands:** Periodic state returns
* **Isolated pixels:** Mixing quality

### 2. State-of-the-Art Results

#### (a) Ascon Results

| Round | Method / Source | Accuracy | Data Complexity |
| :--- | :--- | :--- | :--- |
| **3** | Prior SOTA | 98.61% | $10^{6}$ |
| | **Proposed (m = 1)** | **99.99%** | **$2^{17}$** |
| | **Proposed (m = 2)** | **99.35%** | **$2^{17}$** |
| **4** | Prior SOTA | 50.69% | $10^{7}$ |
| | **Proposed (m = 1)** | **51.22%** | **$2^{17}$** |
| | **Proposed (m = 2)** | **51.39%** | **$2^{17}$** |

#### (b) Speck32/64 Results

| Round | Method / Source | Accuracy | Data Complexity |
| :--- | :--- | :--- | :--- |
| **7** | Prior SOTA (1) | 61.70% | $10^7$ |
| | Prior SOTA (2) | 53.13% | $10^6$ |
| | **Proposed (m = 1)** | **62.01%** | **$2^{17}$** |
| | **Proposed (m = 2)** | **61.98%** | **$2^{17}$** |
| **8** | Prior SOTA | 51.40% | $10^6$ |
| | **Proposed (m = 1)** | **52.09%** | **$2^{17}$** |
| | **Proposed (m = 2)** | **52.00%** | **$2^{17}$** |


Standard **ResNet{18,34,50,101,152} / DenseNet{121,169,201,264}** achieve SOTA via texture representation

### 3. Three-Phase Framework
