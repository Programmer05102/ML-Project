# 🧠 AI Object Recognition System using KNN, HOG & PCA

A complete end-to-end Machine Learning project that automatically collects datasets, trains an optimized image classification model, and performs real-time object recognition through a webcam.

This project demonstrates the entire machine learning workflow:

**Data Collection → Data Cleaning → Feature Engineering → Model Optimization → Real-Time Deployment**

---

# ✨ Features

## 📥 Smart Dataset Collection

Choose how you want to build your dataset:

### Option 1 — Web Scraping

Download images automatically from:

* Google Images
* Bing Images

### Option 2 — Webcam Collection

Capture your own training images using your webcam.

### Option 3 — Hybrid Collection (Recommended)

Combine:

* Google Images
* Bing Images
* Webcam Images

for maximum dataset diversity and improved model performance.

---

## 🧹 Automatic Dataset Cleaning

The system automatically:

✅ Removes corrupted images

✅ Detects and removes duplicate images

✅ Organizes data into class folders

---

## 🧠 Machine Learning Pipeline

The project uses traditional Computer Vision and Machine Learning techniques:

### Feature Extraction

Histogram of Oriented Gradients (HOG)

Extracts:

* Edges
* Shapes
* Object structure
* Texture information

while remaining robust to lighting changes.

### Feature Scaling

StandardScaler

Normalizes features before training.

### Dimensionality Reduction

Principal Component Analysis (PCA)

Benefits:

* Faster training
* Faster prediction
* Reduced memory usage
* Less noise

The model automatically retains **95% of feature variance**.

### Classification

K-Nearest Neighbors (KNN)

Optimized using GridSearchCV.

---

# 🔬 Model Optimization

The training script automatically searches for the best parameters:

```python
parameters = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["distance"],
    "knn__metric": ["cosine", "euclidean"]
}
```

The best configuration is selected automatically using 5-Fold Cross Validation.

---

# 🎥 Real-Time Object Recognition

After training, the system can recognize objects live through a webcam.

### Displayed Information

✅ Predicted Object

✅ Confidence Score

✅ FPS (Frames Per Second)

✅ Top-3 Predictions

✅ Unknown Object Detection

---

## Example Output

```text
Object: Bottle

Confidence: 92.7%

FPS: 28

Top Predictions:

🥇 Bottle (92.7%)
🥈 Cup (4.2%)
🥉 Can (2.1%)
```

---

# 🚀 Project Workflow

```text
Dataset Collection
        │
        ▼
Image Cleaning
        │
        ▼
Feature Extraction (HOG)
        │
        ▼
Feature Scaling
        │
        ▼
PCA (95% Variance Retained)
        │
        ▼
KNN Optimization (GridSearchCV)
        │
        ▼
Model Evaluation
        │
        ▼
Model Saving
        │
        ▼
Real-Time Webcam Recognition
```

---

# 📂 Project Structure

```text
project/

├── dataset/
│   ├── bottle/
│   ├── cat/
│   ├── phone/
│   └── ...

├── collect_data.py

├── train_model.py

├── predict.py

├── object_model.pkl

├── classes.pkl

├── model_info.txt

└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Programmer05102/ML-Project

cd ML-Project
```

---

## Install Dependencies

```bash
pip install opencv-python

pip install numpy

pip install scikit-learn

pip install scikit-image

pip install pillow

pip install imagehash

pip install icrawler
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

---

# 📥 Creating a Dataset

Run:

```bash
python collect_data.py
```

Enter object name:

```text
Bottle
```

Choose:

```text
1. Google + Bing Scraper

2. Webcam Collection

3. Both (Recommended)
```

The dataset will automatically be stored inside:

```text
dataset/bottle/
```

---

# 🏋️ Training the Model

Run:

```bash
python train_model.py
```

The training script automatically:

* Loads all classes
* Extracts HOG features
* Applies StandardScaler
* Applies PCA
* Optimizes KNN
* Evaluates performance
* Saves the trained model

---

## Generated Files

### object_model.pkl

Trained machine learning model.

### classes.pkl

Class labels.

### model_info.txt

Training summary containing:

* Total Classes
* Total Images
* Training Samples
* Testing Samples
* Best Cross Validation Accuracy
* Test Accuracy
* Best Hyperparameters

---

# 🎯 Running Real-Time Recognition

Run:

```bash
python predict.py
```

The webcam will open automatically.

Press:

```text
Q
```

to quit.

---

# 📊 Evaluation Metrics

The training script reports:

### Cross Validation Accuracy

Measures model performance during parameter tuning.

### Test Accuracy

Measures performance on unseen data.

### Classification Report

Includes:

* Precision
* Recall
* F1 Score

### Confusion Matrix

Shows class-wise performance.

---

# 💡 Why HOG + PCA + KNN?

This project intentionally uses classical Machine Learning instead of Deep Learning.

Advantages:

✅ Easier to understand

✅ Faster training

✅ Runs on low-end hardware

✅ Great for educational projects

✅ Demonstrates feature engineering concepts

---

# 📈 Expected Performance

Performance depends on:

* Dataset quality
* Number of classes
* Dataset balance
* Image diversity

Typical results:

| Classes | Expected Accuracy |
| ------- | ----------------- |
| 5–10    | 85–95%            |
| 10–20   | 75–90%            |
| 20+     | 65–85%            |

---

# 🔮 Future Improvements

Potential upgrades:

* CNN-Based Classification
* MobileNet Integration
* ResNet Integration
* YOLO Object Detection
* Data Augmentation
* GUI Application
* Confidence Calibration
* Export to Mobile Devices

---

# 🎓 Educational Value

This project demonstrates:

* Data Collection
* Data Cleaning
* Feature Extraction
* Dimensionality Reduction
* Hyperparameter Optimization
* Model Evaluation
* Real-Time Inference
* Computer Vision Fundamentals

making it an excellent Machine Learning, Computer Vision, or Final Year Project portfolio piece.

---

# 🤝 Contributing

Contributions are welcome.

Feel free to:

* Fork the repository
* Create new features
* Report bugs
* Improve performance
* Submit pull requests

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful, consider giving it a star on GitHub.

It helps others discover the project and motivates future improvements.
