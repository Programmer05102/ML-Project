# 🧠 AI Object Recognition using KNN, HOG & PCA

A lightweight real-time object recognition system built with Python, OpenCV, HOG feature extraction, PCA dimensionality reduction, and K-Nearest Neighbors (KNN).

This project automatically collects images from the web, trains an optimized machine learning model, and performs live object recognition using a webcam.

---

## 🚀 Features

✅ Automatic dataset collection from Google and Bing

✅ Duplicate and corrupted image removal

✅ HOG (Histogram of Oriented Gradients) feature extraction

✅ PCA for dimensionality reduction and faster inference

✅ Optimized KNN classifier with Grid Search

✅ Real-time webcam object recognition

✅ Fully automated training pipeline

✅ Supports multiple object categories

---

## 🛠️ Technologies Used

* Python
* OpenCV
* Scikit-Learn
* NumPy
* scikit-image
* PIL (Pillow)
* ImageHash
* iCrawler

---

## 📂 Project Structure

```text
project/
│
├── dataset/
│   ├── cat/
│   ├── dog/
│   ├── bottle/
│   └── ...
│
├── collect_data.py
├── train_model.py
├── predict.py
│
├── object_model.pkl
├── classes.pkl
│
└── README.md
```

---

## 🔍 How It Works

### 1️⃣ Dataset Collection

The system automatically downloads images from Google and Bing using the provided object name.

Example:

```bash
python collect_data.py
```

Input:

```text
Enter object name: bottle
```

The script:

* Downloads hundreds of images
* Removes corrupted files
* Removes duplicate images
* Stores clean data inside the dataset folder

---

### 2️⃣ Feature Extraction

Instead of training directly on raw pixels, the project uses:

**Histogram of Oriented Gradients (HOG)**

HOG captures:

* Edges
* Shapes
* Object structure

while being more robust to lighting and background variations.

---

### 3️⃣ Feature Optimization

The extracted HOG features are processed using:

**PCA (Principal Component Analysis)**

Benefits:

* Reduces feature dimensions
* Removes noise
* Improves training speed
* Improves prediction speed
* Reduces memory usage

---

### 4️⃣ Model Training

The classifier uses:

**K-Nearest Neighbors (KNN)**

GridSearchCV automatically searches for:

* Best K value
* Best distance metric
* Best weighting strategy

Example search space:

```python
n_neighbors = [1, 3, 5, 7, 9, 11, 15, 21, 31]

weights = [
    "uniform",
    "distance"
]

metrics = [
    "euclidean",
    "manhattan",
    "cosine"
]
```

---

### 5️⃣ Real-Time Recognition

After training:

```bash
python predict.py
```

The webcam starts automatically and predicts objects in real time.

Press:

```text
Q
```

to quit.

---

## ⚡ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/object-recognition-knn.git

cd object-recognition-knn
```

### Install Dependencies

```bash
pip install opencv-python

pip install numpy

pip install scikit-learn

pip install scikit-image

pip install pillow

pip install imagehash

pip install icrawler
```

or

```bash
pip install -r requirements.txt
```

---

## 📈 Machine Learning Pipeline

```text
Image
   ↓
Resize (128×128)
   ↓
Histogram Equalization
   ↓
HOG Feature Extraction
   ↓
StandardScaler
   ↓
PCA (95% Variance Retained)
   ↓
KNN Classifier
   ↓
Prediction
```

---

## 🎯 Example Use Cases

* Educational AI Projects
* Computer Vision Learning
* Real-Time Object Detection Demos
* Machine Learning Coursework
* Image Classification Research
* Rapid Prototyping

---

## 📊 Performance

Performance depends on:

* Dataset quality
* Number of classes
* Number of images per class

Typical results:

| Classes | Accuracy |
| ------- | -------- |
| 5–10    | 85–95%   |
| 10–20   | 75–90%   |
| 20+     | 65–85%   |

---

## 🔮 Future Improvements

* CNN-based Deep Learning Models
* MobileNet Integration
* ResNet Integration
* Object Detection with YOLO
* Data Augmentation
* GUI Application
* Model Confidence Scores

---

## 🤝 Contributing

Contributions are welcome.

Feel free to:

* Fork the repository
* Create a feature branch
* Submit a pull request
* Open issues and suggestions

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.
