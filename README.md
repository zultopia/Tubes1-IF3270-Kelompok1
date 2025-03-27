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
3. [Prerequisites](#prerequisites)
4. [Installation and Setup](#installation-and-setup)
5. [Structure Program](#structure-program)

## 📌 Description
Proyek ini merupakan implementasi Feedforward Neural Network (FFNN) dari awal untuk Tugas Besar IF3270 Pembelajaran Mesin. Tujuan utama adalah mengembangkan pemahaman mendalam tentang arsitektur neural network tiruan dengan mengimplementasikan model secara manual tanpa menggunakan library deep learning tingkat tinggi.

## 📂 Features
- Implementasi FFNN dari scratch
- Fungsi Aktivasi : Linear, ReLU, Sigmoid, Hyperbolic Tangent (tanh), dan Softmax
- Fungsi loss : Mean Squared Error (MSE), Binary Cross-Entropy, Categorical Cross-Entropy
- Metode inisialisasi bobot : Zero Initialization, Random Uniform Distribution, Random Normal Distribution

## 🛠️ Prerequisites
Pastikan kamu sudah menginstall dependencies berikut:
- Python 3.8+
- NumPy
- Scikit-learn
- Matplotlib

## 🚀 Installation and Setup
1. Clone repository:
```sh
git clone https://github.com/zultopia/Tubes1-IF3270-Kelompok1.git
cd src
```

2. Buat virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
# atau
venv\Scripts\activate  # Untuk Windows
```

3. Install dependencies:
```sh
pip install -r requirements.txt
```

## 🎯 Structure Program
```sh
src/
│
├── FFNN/
│   ├── Activation.py     
│   ├── Initializer.py              
│   ├── Loss.py
│   └── Model.py      
│
├── test_final.ipynb
│
├── README.md
└── requirements.txt
```

## 📜 License
This project is released under the MIT License.

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/zultopia/Tubes1-IF3270-Kelompok1/blob/main/LICENSE