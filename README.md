# Coherent Learning in Quantum Circuits

Official implementation of **"Coherent Learning in Quantum Circuits"**. TODO: incluir el link de preprint

---

## 📄 Overview

This repository contains the source code for **Coherent Learning in Quantum Circuits**, a project that investigates **coherent training methodologies** for Quantum Neural Networks (QNNs). Key contributions include:

- Implementation of **data re-uploading techniques** for QNNs.  
- Development of **custom quantum gates** and parameterized quantum layers.  
- Integration of **optimization strategies** for efficient training.  
- Inclusion of a **Quantum Annealing Simulator** for testing and benchmarking.

---

## 🧑‍💻 Authors

- **Paper Authors:** Javier Gonzalez Conde, Lirande Pira, Pablo Rodríguez Grasa, and Ignacio B. Acedo.  
- **Code Author:** All source code in this repository was written exclusively by **Ignacio B. Acedo**.

---

## 💻 Python and Julia Implementations

The repository provides both **Python** and **Julia** versions of the codebase:

| Language | Notes |
|----------|-------|
| **Python** | Original implementation. Fully tested, stable, and recommended for most users. |
| **Julia** | AI-translated from Python to leverage native speed. Functionality is **not guaranteed**; review carefully before use. |

⚠️ **Important:** The Julia implementation may contain bugs, inconsistencies, or outdated features relative to the Python version. For a fully tested and up-to-date experience, use Python.

---

## 🚀 Getting Started

### Requirements

- Python ≥ 3.9 (for Python version)  
- Julia ≥ 1.9 (for Julia version)  
- Standard scientific Python packages: `numpy`, `scipy`, etc. *(See `requirements.txt`)*  

### Installation

1. Clone the repository:

```bash
git clone https://github.com/nacedob/coherent-learning-qml.git
cd coherent-learning-qml


2. For Python:
```bash
pip install -r requirements.txt
```
3. For Julia:
```bash
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Runnning examples

- **Python**: 
```bash
python CoherentQMLPython/main.py
```  
- **Julia**:
```bash
julia CoherentQMLJulia/main.jl
```


### Repository Structure

```
coherent-learning-qml/
├── python/          # Original Python implementation
├── julia/           # AI-translated Julia implementation
├── notebooks/       # Example experiments and demos
├── tests/           # Unit tests for Python code
├── README.md
├── requirements.txt # Python dependencies
└── Project.toml     # Julia dependencies
```

### Disclaimer

- The Julia code is provided as-is and may not fully replicate the Python functionality.
- Python is the primary, fully tested version and should be preferred for research or production use.

### Contact

- **Ignacio B. Acedo**: [GitHub](https://github.com/nacedob), [LinkedIn](https://www.linkedin.com/in/ignaciobenitoacedoblanco/), [mail](mailto:iacedo@ucm.es)