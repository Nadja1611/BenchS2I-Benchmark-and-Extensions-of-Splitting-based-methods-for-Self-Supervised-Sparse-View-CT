# BenchS2I

**Benchs2i** is a benchmark dataset and evaluation framework designed to support research in *[briefly state your domain here, e.g., image-to-image translation, simulation-to-image, etc.]*. It provides standardized data, reproducible evaluation protocols, and curated subsets to facilitate fair comparison across methods.

---

## 📦 Dataset Overview

The **LoDoPaB-CT Dataset** used in this project is constructed using ground truth testing data.

You can download the original data here:
👉 https://zenodo.org/records/3384092

### 📌 Data Source

The dataset is derived from publicly available ground truth CT data. We use the **testing split** as the foundation and convert the .hdf5 files to .pt files to build the LoDoPaB-CT dataset.

---



## 📁 Repository Structure

```
Benchs2i/
utils.py
models.py
run.py
sparse2inverse_ds.py
utils_lodopab.py
```

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/Nadja1611/BenchS2I-Benchmark-and-Extensions-of-Splitting-based-methods-for-Self-Supervised-Sparse-View-CT.git
cd Benchs2i
```

### 2. Download the dataset

```
bash scripts/download_data.sh
```

Or manually download from:
https://zenodo.org/records/3384092

---
