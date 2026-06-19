import cv2
import time
import pickle
import numpy as np

from skimage.feature import hog

IMG_SIZE = (64, 64)

# Tune this after testing
UNKNOWN_DISTANCE_THRESHOLD = 1.0


def extract_features(image):

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    ).astype(np.float32)

    return features


print("Loading Model...")

with open(
    "object_model.pkl",
    "rb"
) as f:

    model = pickle.load(f)

print("Model Loaded Successfully.")

# Extract KNN from pipeline
knn = model.named_steps["knn"]

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    feature = extract_features(frame)

    feature = feature.reshape(1, -1)

    # Apply scaler + PCA only
    transformed_feature = model[:-1].transform(
        feature
    )

    # KNN Prediction
    prediction = knn.predict(
        transformed_feature
    )[0]

    probabilities = knn.predict_proba(
        transformed_feature
    )[0]

    confidence = (
        np.max(probabilities) * 100
    )

    # Distance to nearest neighbor
    distances, _ = knn.kneighbors(
        transformed_feature,
        n_neighbors=1
    )

    nearest_distance = distances[0][0]

    # Unknown Detection
    if nearest_distance > UNKNOWN_DISTANCE_THRESHOLD:

        prediction = "Unknown"

    class_names = knn.classes_

    top3_indices = np.argsort(
        probabilities
    )[::-1][:3]

    current_time = time.time()

    fps = 1 / (
        current_time - prev_time
    )

    prev_time = current_time

    # Confidence Color
    if confidence >= 80:

        confidence_color = (
            0,
            255,
            0
        )

    elif confidence >= 60:

        confidence_color = (
            0,
            255,
            255
        )

    else:

        confidence_color = (
            0,
            0,
            255
        )

    cv2.putText(
        frame,
        f"Object: {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        confidence_color,
        2
    )

    cv2.putText(
        frame,
        f"Distance: {nearest_distance:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Top Predictions:",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y = 260

    for rank, idx in enumerate(
        top3_indices,
        start=1
    ):

        label = class_names[idx]

        score = (
            probabilities[idx] * 100
        )

        text = (
            f"{rank}. "
            f"{label} "
            f"({score:.1f}%)"
        )

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y += 35

    cv2.imshow(
        "AI Object Recognition (Press Q to Quit)",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()