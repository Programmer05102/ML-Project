# 🧠 AI Object Recognition System using HOG, PCA & Optimized KNN

A complete end-to-end Computer Vision and Machine Learning project that automatically collects image datasets, trains an optimized object recognition model, and performs real-time object classification through a webcam.

The project demonstrates the entire machine learning workflow:

**Data Collection → Data Cleaning → Feature Engineering → Dimensionality Reduction → Hyperparameter Optimization → Real-Time Object Recognition**

---

# 🚀 Features

### 📥 Automated Dataset Collection

* Download images automatically from:

  * Google Images
  * Bing Images
  * Baidu Images
* Collect custom images directly from webcam
* Combine web scraping and webcam collection for better datasets

### 🧹 Dataset Cleaning

* Corrupted image detection and removal
* Duplicate image detection using Perceptual Hashing (pHash)
* Automatic dataset validation

### 🎯 Feature Engineering

* Image resizing and normalization
* Histogram Equalization
* HOG (Histogram of Oriented Gradients) feature extraction
* Lightweight and fast feature computation

### 📉 Dimensionality Reduction

* Principal Component Analysis (PCA)
* Reduces feature dimensionality
* Faster training and prediction
* Lower memory consumption

### 🤖 Optimized KNN Classifier

* GridSearchCV hyperparameter optimization
* Automatic selection of:

  * Number of Neighbors (K)
  * Distance Metric
  * Weighting Strategy
* Cosine similarity based classification

### 🎥 Real-Time Object Recognition

* Webcam-based prediction
* FPS display
* Top-3 predictions display
* Confidence score display
* Distance-based Unknown Object Detection
* ROI (Region of Interest) object recognition

### 🚫 Unknown Object Detection

Unlike traditional KNN implementations that always predict a known class, this system measures the nearest-neighbor distance and rejects objects that are too different from the training dataset.

Example:

Known Object:
Chair → Chair

Unknown Object:
Mobile Phone → Unknown

This makes the system significantly more practical for real-world usage.

---

# 📂 Project Structure

```text
project/
│
├── dataset/
│   ├── chair/
│   ├── table/
│   ├── fridge/
│   ├── tv/
│   └── almirah/
│
├── collect_data.py
├── train_model.py
├── predict.py
│
├── object_model.pkl
├── classes.pkl
├── model_info.txt
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

### Programming Language

* Python 3

### Computer Vision

* OpenCV

### Machine Learning

* Scikit-Learn

### Feature Extraction

* Scikit-Image (HOG)

### Dataset Collection

* iCrawler

### Image Processing

* ImageHash

### Numerical Computing

* NumPy

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/Programmer05102/ML-Project.git

cd ML-Project
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📥 Step 1 – Collect Dataset

Run:

```bash
python collect_data.py
```

Example:

```text
Enter object name: Chair

Select Data Collection Method

1. Google + Bing + Baidu Scraper
2. Webcam Collection
3. Both (Recommended)
```

Recommended dataset size:

| Object Count | Accuracy Potential |
| ------------ | ------------------ |
| 100 Images   | Low                |
| 300 Images   | Good               |
| 500 Images   | Very Good          |
| 1000+ Images | Excellent          |

---

# 🧠 Step 2 – Train Model

Run:

```bash
python train_model.py
```

Training pipeline:

```text
Images
   ↓
Resize
   ↓
Histogram Equalization
   ↓
HOG Features
   ↓
StandardScaler
   ↓
PCA
   ↓
KNN
```

The system automatically:

* Loads dataset
* Extracts features
* Splits train/test data
* Performs Grid Search
* Evaluates accuracy
* Saves trained model

Generated files:

```text
object_model.pkl
classes.pkl
model_info.txt
```

---

# 🎥 Step 3 – Real-Time Prediction

Run:

```bash
python predict.py
```

The webcam opens and displays:

```text
Object: Chair
Confidence: 91.4%
Distance: 0.42
FPS: 24

Top Predictions:

1. Chair (91.4%)
2. Table (6.3%)
3. TV (2.3%)
```

---

# 🎯 Region of Interest (ROI)

To improve recognition accuracy, prediction is performed only on the center ROI instead of the entire webcam frame.

Without ROI:

```text
Face + Wall + Fan + Object
```

With ROI:

```text
Object Only
```

Benefits:

* Better classification accuracy
* Less background noise
* More stable predictions
* Improved Unknown detection

---

# 📊 Example Training Results

Dataset:

```text
Almirah : 200 images
Chair   : 200 images
Fridge  : 150 images
Table   : 100 images
TV      : 150 images
```

Results:

```text
Cross Validation Accuracy : 94.38%

Test Accuracy : 95.62%
```

Confusion Matrix:

```text
Accurate separation between all 5 object classes.
```

---

# 🔬 Machine Learning Concepts Used

This project demonstrates:

* Supervised Learning
* Image Classification
* Feature Extraction
* HOG Features
* PCA
* K-Nearest Neighbors
* Hyperparameter Tuning
* Cross Validation
* Model Evaluation
* Real-Time Computer Vision
* Unknown Class Detection

---

# 💡 Future Improvements

Possible upgrades:

### Deep Learning

* CNN
* MobileNet
* EfficientNet
* ResNet

### Object Detection

* YOLOv8
* SSD
* Faster R-CNN

### Deployment

* Flask API
* Streamlit Web App
* Android Application
* Raspberry Pi Deployment

### Dataset Enhancements

* Automatic Background Removal
* Data Augmentation
* Object Segmentation

---

# 🎓 Educational Value

This project is ideal for:

* Machine Learning Beginners
* Computer Vision Projects
* Final Year Projects
* BSCS Students
* AI Portfolio Projects
* Internship Demonstrations

It showcases the complete ML lifecycle from data collection to real-time deployment.

---

# 📜 License

This project is open-source and available under the MIT License.

---

# 👨‍💻 Author

Developed as a Machine Learning and Computer Vision project using Python, OpenCV, Scikit-Learn, HOG, PCA, and Optimized KNN.
