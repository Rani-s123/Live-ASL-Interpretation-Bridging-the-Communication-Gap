# Real-Time American Sign Language (ASL) Fingerspelling Recognition using CNN

## 📝 Abstract

American Sign Language (ASL) is one of the most natural and expressive forms of communication used by the Deaf and Mute (D&M) community. However, due to a lack of widespread understanding and limited availability of interpreters, communication can be challenging. This project presents a **real-time ASL fingerspelling recognition system** using **Convolutional Neural Networks (CNNs)** that identifies static hand gestures corresponding to the 26 English alphabets with **98.0% accuracy**.

---

## 📌 Project Description

### 👋 Motivation

Communication is a fundamental human need. D&M individuals rely on hand gestures for daily communication, but the barrier arises when interacting with those unfamiliar with sign language. Our goal is to bridge this gap using a computer vision-based solution that can interpret ASL fingerspelling in real time.

---

## 💡 Key Features

- Real-time hand gesture recognition using webcam input.
- ROI (Region of Interest) detection and preprocessing using Gaussian blur and adaptive thresholding.
- Custom CNN model trained on a self-generated dataset.
- GUI for live translation of signs into text.
- Multiple classifiers for improved accuracy in ambiguous gesture classification.

---

## 🔧 Tech Stack

- **Programming Language**: Python 3.8+
- **Libraries**:
  - OpenCV
  - TensorFlow
  - Keras
  - NumPy
  - Pillow
  - Tkinter
  - pyenchant & cyhunspell (for spell-check and suggestions)

---

## 🛠️ Steps to Build the Project

### 1️⃣ Dataset Preparation

- Created a custom dataset with separate folders for each alphabet (A-Z) under `trainingData` and `testingData`.
- Images captured from webcam with a defined **Region of Interest (ROI)**.
- Applied preprocessing: grayscale conversion → Gaussian blur → adaptive thresholding.

```python
# Example code snippet for preprocessing
import cv2

def preprocess_image(path):
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    _, result = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return result
```

---

### 2️⃣ CNN Model Architecture

- **Input Layer**: 128x128 grayscale images.
- **Convolutional Layers**: Detect local patterns.
- **Pooling Layers**: Reduce dimensionality (Max & Average Pooling).
- **Fully Connected Layers**: Classify based on extracted features.
- **Output Layer**: 26 nodes (A–Z), softmax activation.

**Loss Function**: Cross-entropy  
**Optimizer**: Adam (Adaptive Moment Estimation)

---

### 3️⃣ Handling Gesture Ambiguities

Certain gestures are visually similar and may be misclassified. To address this:

- Built **three specialized classifiers** for subsets of confusing gestures:
  - `{D, R, U}`
  - `{T, K, D, I}`
  - `{S, M, N}`

This two-layer classification approach improved accuracy from **95.8% → 98.0%**.

---

### 4️⃣ GUI Application

- A user-friendly GUI built with **Tkinter** allows:
  - Real-time webcam input
  - Gesture prediction display
  - Text formation from sequences of signs

---

## 📈 Results

- **Accuracy (Single Classifier)**: 95.8%
- **Accuracy (Multi-Classifier Ensemble)**: 98.0%
- Displayed results using **confusion matrices** to evaluate performance per alphabet.

---

## 🔄 Project Flow

### 🧠 Gesture Classification Flowchart

```
Webcam → ROI Extraction → Preprocessing → CNN Classifier(s) → Text Output
```

---

### 📱 Application Diagram

```
User Gestures → Camera → Preprocessing → Classifier → Predicted Character → Sentence Formation → GUI Output
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher

### Required Packages

```bash
pip install --upgrade pip
pip install numpy opencv-python tensorflow keras Pillow tk pyenchant cyhunspell
```

---

## 🚀 Running the Application

```bash
python /path/to/Application.py
```

> Replace `/path/to/` with the actual path where the Python file resides.

---

## 🙌 Final Thoughts

This project demonstrates a practical solution for real-time ASL gesture recognition using deep learning, helping bridge the communication gap between D&M individuals and the rest of society.

