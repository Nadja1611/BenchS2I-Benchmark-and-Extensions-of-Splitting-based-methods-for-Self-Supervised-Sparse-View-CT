# BenchS2I

**BenchS2I** is a benchmark and research framework for evaluating and extending **splitting-based self-supervised learning methods for sparse-view CT reconstruction**. The repository provides implementations of angular splitting (S2I), detector splitting (P2P), and joint angular-detector splitting approaches, together with standardized experimental settings based on the LoDoPaB-CT dataset.

---

## 📦 Dataset Overview

The experiments in this repository are based on the **LoDoPaB-CT Dataset**.

You can download the original dataset here:
👉 https://zenodo.org/records/3384092

### 📌 Data Source

The LoDoPaB-CT dataset consists of simulated low-dose CT measurements derived from clinical CT scans. We use the publicly available data for training and evaluation of sparse-view CT reconstruction methods.

For our experiments, the data is converted from the original `.hdf5` format to `.pt` files for efficient loading during training and evaluation.

---

## 🧩 Implemented Methods

### S2I (Angular Splitting)

The S2I method partitions projection angles into complementary subsets and learns image reconstruction in a self-supervised manner.

### P2P (Detector/Lattice Splitting)

The P2P method partitions detector measurements into complementary subsets and performs self-supervised reconstruction using detector-domain splitting.

### DoubleSplit

An extension that combines both angular splitting and detector splitting during training, enabling joint self-supervision from projection angles and detector measurements.

---

## 📁 Repository Structure

```text
BenchS2I/
├── run.py                  # Training for S2I and P2P methods
├── run_doublesplit.py      # Training for joint angular + detector splitting
├── inference.py            # Inference and evaluation
├── models.py               # Model architectures
├── utils.py                # General utilities
├── utils_lodopab.py        # LoDoPaB-specific utilities
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nadja1611/BenchS2I-Benchmark-and-Extensions-of-Splitting-based-methods-for-Self-Supervised-Sparse-View-CT.git
cd BenchS2I-Benchmark-and-Extensions-of-Splitting-based-methods-for-Self-Supervised-Sparse-View-CT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the LoDoPaB-CT dataset from:

https://zenodo.org/records/3384092

and prepare the data according to your local setup.

---

## 🏋️ Training

### Angular Splitting (S2I)

Example:

```bash
python run.py \
    --method S2I \
    --angles 16 \
    --number_training_imgs 1000
```

---

### Detector Splitting (P2P)

Example:

```bash
python run.py \
    --method P2P \
    --angles 16 \
    --number_training_imgs 1000
```

---

### Joint Angular + Detector Splitting (DoubleSplit)

Example:

```bash
python run_doublesplit.py \
    -l MSE_data \
    -grid_size 2 \
    --angles 16 \
    --number_training_imgs 1000 \
    --method S2I
```

---

## 🔊 Correlated Noise Experiments

To enable correlated noise, add:

```bash
--correlated_noise
```

Example:

```bash
python run_doublesplit.py \
    -l MSE_data \
    -grid_size 2 \
    --angles 16 \
    --correlated_noise \
    --number_training_imgs 1000 \
    --method S2I
```

---

## 🔄 Interpolation

For correlated-noise experiments, interpolation can be enabled using:

```bash
-i
```

Example:

```bash
python run_doublesplit.py \
    -l MSE_data \
    -grid_size 2 \
    --angles 16 \
    --correlated_noise \
    --number_training_imgs 1000 \
    -i \
    --method S2I
```

To run without interpolation, simply omit the `-i` flag.

---

## 📊 Main Parameters

| Parameter | Description |
|------------|------------|
| `--method S2I` | Angular splitting |
| `--method P2P` | Detector/lattice splitting |
| `--angles` | Number of projection angles |
| `--number_training_imgs` | Number of training images |
| `--correlated_noise` | Enable correlated noise |
| `-i` | Enable interpolation |
| `-grid_size` | Detector splitting grid size |
| `-l` | Loss function |

---
