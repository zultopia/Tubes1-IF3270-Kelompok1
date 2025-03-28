<br />
<div align="center">
  <h1 align="center">Tugas Besar 1 IF3270 : Implementasi Feedforward Neural Network (FFNN)</h1>

   <br />
    <a href="https://github.com/zultopia/Tubes1-IF3270-Kelompok1.git">Report Bug</a>
    ·
    <a href="https://github.com/zultopia/Tubes1-IF3270-Kelompok1.git">Request Feature</a>
    <br>
<br>

[![MIT License][license-shield]][license-url]

  </p>
</div>

<div align="center" id="contributor">
  <strong>
    <h3>Created by Kelompok 1:</h3>
    <table align="center">
      <tr>
        <td>Name</td>
        <td>NIM</td>
      </tr>
      <tr>
        <td>Marzuli Suhada M</td>
         <td>13522070</td>
     </tr>
     <tr>
        <td>Ahmad Mudabbir Arif</td>
         <td>13522072</td>
    </tr>
     <tr>
        <td>Naufal Adnan</td>
         <td>13522116</td>
    </tr>
    </table>
  </strong>
</div>

## Table of Contents
1. [Description](#description)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation and Setup](#installation-and-setup)
5. [Project Structure](#project-structure)
6. [How to Run](#how-to-run)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
7. [Team Contributions](#team-contributions)
8. [License](#license)

---

## Description
Proyek ini merupakan implementasi **Feedforward Neural Network (FFNN)** dari scratch untuk **Tugas Besar IF3270 Pembelajaran Mesin**.  

Tujuan utama dari proyek ini adalah mengembangkan pemahaman mendalam tentang arsitektur **neural network** dengan mengimplementasikan model secara manual tanpa menggunakan library deep learning tingkat tinggi.

---

## Features
- Implementasi **FFNN dari scratch**.
- Mendukung berbagai **fungsi aktivasi**:
  - Linear
  - ReLU
  - Sigmoid
  - Hyperbolic Tangent (tanh)
  - Softmax
  - **Leaky ReLU** *(Bonus)*
  - **ELU** *(Bonus)*
  - **Swish** *(Bonus)*
- Mendukung berbagai **fungsi loss**:
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
- **Metode inisialisasi bobot** yang didukung:
  - Zero Initialization
  - Random Uniform Distribution
  - Random Normal Distribution
  - **Xavier Initialization** *(Bonus)*
  - **He Initialization** *(Bonus)*
- **Regularisasi** *(Bonus)*:
  - Implementasi metode **L1 dan L2 Regularization**.
  - Eksperimen dengan model tanpa regularisasi, dengan L1, dan dengan L2.
  - Analisis perbandingan hasil prediksi, grafik loss pelatihan, serta distribusi bobot dan gradien bobot.
- **Normalisasi** *(Bonus)*:
  - Implementasi metode **RMSNorm**.
  - Eksperimen dengan model tanpa normalisasi dan dengan normalisasi.
  - Analisis perbandingan hasil prediksi, grafik loss pelatihan, serta distribusi bobot dan gradien bobot.

---

## System Requirements
- **Python** 3.8+
- **Libraries**:
  - NumPy
  - Scikit-learn
  - Matplotlib (untuk visualisasi)
  - dash
  - dash_cytoscape

---

## Installation and Setup

1. **Clone repository**  
   ```bash
   git clone https://github.com/zultopia/Tubes1-IF3270-Kelompok1.git
   cd Tubes1-IF3270-Kelompok1
   ```

2. **Buat virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    # atau
    venv\Scripts\activate  # Untuk Windows
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
---

## Project Structure
    TUBES1-IF3270-KELOMPOK1/
    │
    ├── src/
    │   └── FFNN/
    │       ├── __init__.py
    │       ├── Activation.py
    │       ├── Initializer.py
    │       ├── Loss.py
    │       └── Model.py
    │
    ├── tests/
    │   └── test_final.ipynb
    │
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt

---

## How to Run

Buat file **.ipynb** di dalam folder src. Buat skrip codenya lalu jalankan.

### Training the Model

Contoh skrip untuk melatih model FFNN dengan dataset yang telah diproses:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from FFNN.Model import FFNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model setup
layers = [X_train.shape[1], 128, 64, 10]
activations = ['relu', 'relu', 'softmax']
loss = 'categorical_cross_entropy'

model = FFNN(layers, activations, loss, init_method='xavier', seed=42)
history = model.train(X_train, y_train, epochs=15, lr=0.01, batch_size=128)

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()
```

### Evaluating the Model

Setelah model dilatih, lakukan evaluasi performanya dengan menggunakan dataset uji:

```python
# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

---

## Team Contributions

| Team Member | Responsibilities |
|-------------|----------------|
| **Naufal Adnan** | **Implementasi:** <br> - Setup inputan jumlah neuron <br> - Implementasi fungsi aktivasi dan turunan Linear, ReLU <br> - Implementasi Loss Function dan turunan Binary Cross Entropy & Categorical Cross Entropy <br> - Implementasi method untuk menampilkan model berupa struktur jaringan beserta bobot dan gradien bobot tiap neuron dalam bentuk graf <br> - Instance model yang diinisialisasikan harus bisa menyimpan bobot & harus bisa menyimpan gradien bobot <br> - Implementasi forward propagation dengan batch input <br> - Implementasi proses pelatihan dengan batch size, learning rate, jumlah epoch, dan verbose <br> - Mengembalikan histori pelatihan (training loss & validation loss) <br> - Implementasi metode regularisasi L1 dan L2 <br> - Implementasi metode normalisasi RMSNorm <br> **Pengujian:** <br> - Pengaruh depth dan width <br> - Pengaruh fungsi aktivasi <br> **Laporan** |
| **Ahmad Mudabbir Arif** | **Implementasi:** <br> - Implementasi fungsi aktivasi dan turunan Sigmoid & Hyperbolic <br> - Implementasi Loss Function dan turunan MSE <br> - Implementasi method untuk menampilkan distribusi bobot dari tiap layer <br> - Implementasi method untuk menampilkan distribusi gradien bobot dari tiap layer <br> - Instance model yang diinisialisasikan harus bisa menyimpan bobot & harus bisa menyimpan gradien bobot <br> - Implementasi weight update dengan gradient descent <br> - Implementasi proses pelatihan dengan batch size, learning rate, jumlah epoch, dan verbose <br> - Mengembalikan histori pelatihan (training loss & validation loss) <br> - Implementasi 2 metode inisialisasi bobot *(Bonus)* <br> **Pengujian:** <br> - Pengaruh learning rate <br> - Pengaruh inisialisasi bobot <br> **Laporan** |
| **Marzuli Suhada M** | **Implementasi:** <br> - Implementasi fungsi aktivasi dan turunan Softmax <br> - Implementasi inisialisasi bobot untuk Zero Initialization, Random Uniform, Random Normal <br> - Implementasi method untuk save dan load <br> - Instance model yang diinisialisasikan harus bisa menyimpan bobot & harus bisa menyimpan gradien bobot <br> - Implementasi backward propagation dengan chain rule untuk menghitung gradien <br> - Implementasi proses pelatihan dengan batch size, learning rate, jumlah epoch, dan verbose <br> - Mengembalikan histori pelatihan (training loss & validation loss) <br> - Implementasi minimal 2 fungsi aktivasi lain yang sering digunakan <br> **Pengujian:** <br> - Pengaruh regularisasi (jika mengerjakan) <br> - Pengaruh normalisasi RMSNorm (jika mengerjakan) <br> - Perbandingan dengan library Sklearn <br> **Laporan** |

---

## 📜 License
This project is released under the MIT License.

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/zultopia/Tubes1-IF3270-Kelompok1/blob/main/LICENSE