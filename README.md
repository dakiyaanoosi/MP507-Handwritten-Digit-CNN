# MP507 - Handwritten Digit Recognition using CNN
**B.Tech. 3rd Year Mini Project**  
*Department of Computer Science & Engineering, UNSIET, Veer Bahadur Singh Purvanchal University, Jaunpur*

---

## 📌 Project Overview
MP507 is an end-to-end, production-ready Handwritten Digit Recognition web platform. It pairs a regularized 4-convolution Batch-Normalized Convolutional Neural Network (CNN) trained on the MNIST benchmark with an OpenCV spatial normalization engine (20×20 bounding box + center-of-mass translation) and a responsive Flask web server.

### Key Features
- **CNN Architecture:** 4-Conv layers with Batch Normalization, Max Pooling, and Dropout (469,098 parameters).
- **Data Augmentation:** Native Keras 3 layers (`RandomRotation`, `RandomTranslation`, `RandomZoom`).
- **Spatial Normalization Pipeline:** OpenCV bounding-box cropping, proportional 20×20 scaling, and center-of-mass affine translation to (14, 14) matching MNIST standards.
- **Interactive UI:** 280×280 HTML5 drawing canvas with mouse and touch support, file upload module with polarity auto-detection, and real-time 0–9 probability distribution bars.
- **RESTful API:** Clean `/api/predict` and `/api/health` endpoints returning top-1 prediction, confidence score, full 10-class probability array, and latency in milliseconds.

---

## 📊 Experimental Results & Benchmarks

### Local Evaluation Summary (Table 8)
| Evaluation Metric | Observed Value (Local Training Run) |
| :--- | :--- |
| **Best Validation Accuracy & Epoch** | **99.52%** at Epoch 7 |
| **Held-Out Test Accuracy (10,000 samples)** | **99.32%** (0.68% error rate) |
| **Held-Out Test Loss** | **0.0223** |
| **Macro-Average F1-Score** | **0.9931** |
| **Most Frequently Confused Digit Pair** | **5 ↔ 3** (Only 6 misclassifications in 10,000 samples) |
| **Warm Single-Image API Inference Latency** | **67.91 ms** (Target: ≤ 100 ms) |

---

## 📁 Repository Structure
```
OCR/
├── app.py                      # Flask application factory, WSGI runner & startup model loader
├── routes.py                   # API routes (/, /api/health, /api/predict) & OpenCV preprocessor
├── train_cnn.py                # CNN training loop, Keras 3 augmentation, metrics logger & .keras export
├── requirements.txt            # Pinned dependencies
├── test_system.py              # Unit & integration test suite (AST syntax, preprocessing, endpoints)
├── generate_report_assets.py   # Matplotlib high-resolution plot generator for academic report
├── models/
│   ├── mnist_cnn_model.keras   # Checkpointed trained model weights (469,098 parameters)
│   └── training_metrics.json   # Exported test loss, accuracy, confusion matrix, epoch history
├── templates/
│   └── index.html              # Single-file responsive HTML5 Canvas, file upload & CSS3/Vanilla JS UI
└── docs/
    ├── plots/                  # Generated 300 DPI figures for Chapter 6 of the project report
    │   ├── fig1_2_mnist_samples.png
    │   ├── fig6_1_1_class_distribution.png
    │   ├── fig6_1_2_preprocessed_grid.png
    │   ├── fig6_2_1_accuracy_curve.png
    │   ├── fig6_2_2_loss_curve.png
    │   ├── fig6_3_confusion_matrix.png
    │   └── fig6_5_2_preprocessing_pipeline.png
    └── screenshots/            # UI captures
```

---

## 🚀 Quick Setup & Execution Guide

### 1. Prerequisites
- Python 3.10 – 3.13 (64-bit)
- Pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python test_system.py
```

### 4. (Optional) Re-Train CNN Model
```bash
python train_cnn.py
```
*(The trained model `models/mnist_cnn_model.keras` and metrics `models/training_metrics.json` are already included).*

### 5. Start the Web Server Prototype
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 6. Generate Figures for Academic Report
To generate all 300 DPI publication-ready figures for Chapter 6:
```bash
python generate_report_assets.py
```
The figures will be stored in `docs/plots/`.
