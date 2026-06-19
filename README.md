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
- Google Images
- Bing Images

### Option 2 — Webcam Collection

Capture your own training images using your webcam.

### Option 3 — Hybrid Collection (Recommended)

Combine:
- Google Images
- Bing Images
- Webcam Images

for maximum dataset diversity and improved model performance.

---

## 🧹 Automatic Dataset Cleaning

The system automatically:

- Removes corrupted images  
- Detects and removes duplicate images  
- Organizes data into class folders  

---

## 🧠 Machine Learning Pipeline

The project uses traditional Computer Vision and Machine Learning techniques:

### Feature Extraction

**Histogram of Oriented Gradients (HOG)** extracts:
- Edges  
- Shapes  
- Object structure  
- Texture information  

while remaining robust to lighting changes.

---

### Feature Scaling

**StandardScaler** normalizes features before training.

---

### Dimensionality Reduction

**Principal Component Analysis (PCA)** improves efficiency:

- Faster training  
- Faster prediction  
- Reduced memory usage  
- Less noise  

The model retains **95% of feature variance**.

---

### Classification

**K-Nearest Neighbors (KNN)** optimized using GridSearchCV.

---

# 🔬 Model Optimization

The system automatically searches for the best hyperparameters:

```python
parameters = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["distance"],
    "knn__metric": ["cosine", "euclidean"]
}
````

Best configuration is selected using **5-Fold Cross Validation**.

---

# 🎥 Real-Time Object Recognition

After training, the system performs live object recognition via webcam.

## Displayed Information

* Predicted Object
* Confidence Score
* FPS (Frames Per Second)
* Top-3 Predictions
* Unknown Object Detection

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
pip install ImageHash
pip install icrawler
pip install opencv-python
pip install scikit-learn
pip install scikit-image
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

# 📥 Creating a Dataset

```bash
python collect_data.py
```

Enter object name:

```text
Bottle
```

Choose dataset source:

```text
1. Google + Baidu + Bing Scraper
2. Webcam Collection
3. Both (Recommended)
```

Dataset will be stored in:

```text
dataset/bottle/
```

---

# 🏋️ Training the Model

```bash
python train_model.py
```

The script automatically:

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

Training summary including:

* Total Classes
* Total Images
* Training Samples
* Testing Samples
* Best Accuracy
* Best Hyperparameters

---

# 🎯 Running Real-Time Recognition

```bash
python predict.py
```

Press:

```text
Q
```

to quit.

---

# 📊 Evaluation Metrics

* Cross Validation Accuracy
* Test Accuracy
* Precision / Recall / F1-score
* Confusion Matrix

---

# 💡 Why HOG + PCA + KNN?

This project intentionally avoids Deep Learning to focus on fundamentals.

Advantages:

* Easy to understand
* Fast training
* Works on low-end hardware
* Strong for learning ML concepts
* Great academic/project value

---

# 📈 Expected Performance

| Classes | Accuracy |
| ------- | -------- |
| 5–10    | 85–95%   |
| 10–20   | 75–90%   |
| 20+     | 65–85%   |

---

# 🔮 Future Improvements

* CNN-Based Classification
* MobileNet / ResNet Integration
* YOLO Object Detection
* Data Augmentation
* GUI Application
* Mobile Deployment

---

# 🎓 Educational Value

This project demonstrates:

* Data Collection
* Data Cleaning
* Feature Engineering
* Dimensionality Reduction
* Hyperparameter Tuning
* Model Evaluation
* Real-Time Inference

---

# 🤝 Contributing

* Fork the repository
* Add features
* Report issues
* Improve performance
* Submit pull requests

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you find this useful, consider starring the repository.