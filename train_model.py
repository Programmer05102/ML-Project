import os
import cv2
import pickle
import numpy as np

from skimage.feature import hog

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATASET_PATH = "dataset"
MODEL_PATH = "object_model.pkl"
INFO_PATH = "model_info.txt"

IMG_SIZE = (64, 64)


def extract_features(image):

    image = cv2.resize(image, IMG_SIZE)

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


print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

X = []
y = []
classes = []

for label in os.listdir(DATASET_PATH):

    class_folder = os.path.join(
        DATASET_PATH,
        label
    )

    if not os.path.isdir(class_folder):
        continue

    classes.append(label)

    count = 0

    for image_name in os.listdir(class_folder):

        image_path = os.path.join(
            class_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        try:

            feature = extract_features(image)

            X.append(feature)
            y.append(label)

            count += 1

        except Exception as e:

            print(
                f"Error processing {image_path}"
            )

            print(e)

    print(f"{label}: {count} images")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nTotal Images:", len(X))
print("Classes:", sorted(classes))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=300, svd_solver="randomized", random_state=42)),
    ("knn", KNeighborsClassifier())
])


parameters = {
    "knn__n_neighbors": [
        3,
        5,
        7,
        9,
        11,
        15,
        21
    ],

    "knn__weights": [
        "distance"
    ],

    "knn__metric": [
        "cosine"
    ]
}


print("\nSearching Best Parameters...")

grid = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print(
    f"\nBest Cross Validation Accuracy: "
    f"{grid.best_score_ * 100:.2f}%"
)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 50)
print("TEST RESULTS")
print("=" * 50)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(
    f"\nModel saved as "
    f"'{MODEL_PATH}'"
)

with open("classes.pkl", "wb") as file:
    pickle.dump(
        sorted(classes),
        file
    )

print(
    "Classes saved as "
    "'classes.pkl'"
)


with open(INFO_PATH, "w") as f:

    f.write("MODEL INFORMATION\n")
    f.write("=" * 40 + "\n\n")

    f.write(
        f"Total Classes: "
        f"{len(classes)}\n"
    )

    f.write(
        f"Total Images: "
        f"{len(X)}\n"
    )

    f.write(
        f"Training Samples: "
        f"{len(X_train)}\n"
    )

    f.write(
        f"Testing Samples: "
        f"{len(X_test)}\n"
    )

    f.write(
        f"\nBest CV Accuracy: "
        f"{grid.best_score_ * 100:.2f}%\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"\nBest Parameters:\n"
    )

    for k, v in grid.best_params_.items():
        f.write(f"{k}: {v}\n")

print(
    f"Model information saved as "
    f"'{INFO_PATH}'"
)

print("\nTraining Completed Successfully!")