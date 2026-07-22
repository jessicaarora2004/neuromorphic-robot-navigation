# Event-Driven Robot Navigation using Spiking Neural Networks

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-red)
![snnTorch](https://img.shields.io/badge/snnTorch-orange)
![Neuromorphic Computing](https://img.shields.io/badge/Neuromorphic-Computing-success)

## Overview

This project explores **neuromorphic computing** for autonomous robot navigation by comparing **Convolutional Neural Networks (CNNs)** with **Spiking Neural Networks (SNNs)**.

The project converts conventional RGB images into temporal spike trains using **rate coding**, processes them using **Leaky Integrate-and-Fire (LIF)** neurons, and predicts one of four navigation commands.

The goal is to study how event-driven neural computation can achieve high accuracy while maintaining sparse neural activity, making it suitable for future low-power robotic systems.

---

##  Objectives

- Build a CNN baseline for robot navigation
- Implement a Spiking Neural Network using snnTorch
- Convert images into spike trains using Rate Coding
- Compare CNN and SNN performance
- Analyze spike sparsity across network layers
- Understand neuromorphic computation for edge robotics

---

##  Project Pipeline

```
Robot Camera Image
        │
        ▼
 Image Preprocessing
        │
        ▼
 Rate Coding
(Image → Spike Train)
        │
        ▼
Spiking Neural Network
(Conv + LIF Layers)
        │
        ▼
Navigation Prediction
        │
        ▼
Forward / Left / Right / Stop
```

---

##  Dataset

Navigation dataset consisting of **13,536 images**.

Classes:

- Forward
- Left
- Right
- Stop

Image Size:

```
64 × 64 × 3
```

---

##  Model Architecture

### CNN Baseline

```
Conv → ReLU → MaxPool
        ↓
Conv → ReLU → MaxPool
        ↓
Conv → ReLU → MaxPool
        ↓
Fully Connected
        ↓
4 Classes
```

---

### Spiking Neural Network

```
Spike Encoding
        ↓
Conv
        ↓
LIF Neuron
        ↓
Conv
        ↓
LIF Neuron
        ↓
Conv
        ↓
LIF Neuron
        ↓
Fully Connected
        ↓
LIF Neuron
        ↓
Output Layer
```

---

## Results

### CNN

| Metric | Value |
|--------|------|
| Accuracy | **100%** |

---

### SNN

| Metric | Value |
|--------|------|
| Accuracy | **100%** |
| Time Steps | 20 |
| Spike Encoding | Rate Coding |

---

## Spike Sparsity Analysis

### Input Spike Sparsity

```
52.47%
```

### Internal Layer Sparsity

| Layer | Sparsity |
|--------|----------|
| LIF1 | 91.11% |
| LIF2 | 91.60% |
| LIF3 | 93.13% |
| LIF4 | 93.70% |

These results demonstrate highly sparse event-driven computation while maintaining perfect classification accuracy.

---

## 🛠 Technologies Used

- Python
- PyTorch
- snnTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

##  Project Structure

```
Project_neuromorphic
│
├── models/
│   ├── cnn.py
│   └── snn.py
│
├── notebooks/
│   ├── 01_perception.ipynb
│   └── 02_robot_navigation.ipynb
│
├── results/
│   └── baseline_results.txt
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/event-driven-robot-navigation-snn.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter

```bash
jupyter notebook
```

Run:

```
01_perception.ipynb
```

followed by

```
02_robot_navigation.ipynb
```

---

## Future Work

- Event Camera Simulation
- Dynamic Vision Sensor (DVS) datasets
- Optical Flow Estimation
- PyBullet Robot Navigation
- STDP Learning
- Reinforcement Learning
- Intel Loihi Deployment
- Real-Time Event-Based Navigation

---

## References

- snnTorch Documentation
- PyTorch Documentation
- Event-Based Vision Literature
- Neuromorphic Computing Research Papers

---
