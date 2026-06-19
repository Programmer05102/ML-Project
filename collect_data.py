import os
import cv2
import imagehash

from PIL import Image
from icrawler.builtin import (
    GoogleImageCrawler,
    BingImageCrawler,
    BaiduImageCrawler
)

MAX_IMAGES_GOOGLE = 300
MAX_IMAGES_BING = 300
MAX_IMAGES_BAIDU = 300


def clean_dataset(dataset_path):

    print("\nChecking corrupted files...")

    for filename in os.listdir(dataset_path):

        filepath = os.path.join(
            dataset_path,
            filename
        )

        try:

            img = Image.open(filepath)
            img.verify()

        except Exception:

            try:

                os.remove(filepath)

                print(
                    "Removed corrupted:",
                    filename
                )

            except Exception as e:

                print(e)

    print("\nRemoving duplicates...")

    hashes = {}
    duplicates = 0

    for filename in os.listdir(dataset_path):

        filepath = os.path.join(
            dataset_path,
            filename
        )

        try:

            image = Image.open(filepath)

            img_hash = imagehash.phash(
                image
            )

            duplicate_found = False

            for existing_hash in hashes:

                if (
                    img_hash - existing_hash
                ) <= 5:

                    duplicate_found = True
                    break

            if duplicate_found:

                os.remove(filepath)
                duplicates += 1

            else:

                hashes[img_hash] = filename

        except Exception:

            pass

    print(
        f"Removed {duplicates} duplicate images."
    )


def crawl_images(
    crawler_class,
    keyword,
    max_num,
    dataset_path
):

    try:

        crawler = crawler_class(
            storage={
                "root_dir": dataset_path
            }
        )

        crawler.crawl(
            keyword=keyword,
            max_num=max_num
        )

    except Exception as e:

        print(
            f"\n{crawler_class.__name__} failed:"
        )

        print(e)


def download_images(
    object_name,
    dataset_path
):

    keyword = f"real {object_name}"

    print(
        f"\nDownloading images for:"
        f" {keyword}"
    )

    print("\nUsing Google...")

    crawl_images(
        GoogleImageCrawler,
        keyword,
        MAX_IMAGES_GOOGLE,
        dataset_path
    )

    print("\nUsing Bing...")

    crawl_images(
        BingImageCrawler,
        keyword,
        MAX_IMAGES_BING,
        dataset_path
    )

    print("\nUsing Baidu...")

    crawl_images(
        BaiduImageCrawler,
        keyword,
        MAX_IMAGES_BAIDU,
        dataset_path
    )

    print("\nDownload complete.")


def collect_from_camera(
    object_name,
    dataset_path
):

    print("\nStarting Camera Collection...")

    print(
        "Press SPACE to save image"
    )

    print(
        "Press Q to finish\n"
    )

    cap = cv2.VideoCapture(0)

    count = len(
        os.listdir(dataset_path)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        cv2.putText(
            display,
            f"Images: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "SPACE=Save | Q=Quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            f"Collecting {object_name}",
            display
        )

        key = cv2.waitKey(1)

        if key == 32:

            filename = os.path.join(
                dataset_path,
                f"cam_{count}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            print(
                "Saved:",
                filename
            )

            count += 1

        elif key == ord("q"):

            break

    cap.release()
    cv2.destroyAllWindows()

    print(
        f"\nCamera collection finished."
    )

    print(
        f"Total images now: {count}"
    )


# -------------------------
# MAIN PROGRAM
# -------------------------

object_name = input(
    "Enter object name: "
).strip()

dataset_path = os.path.join(
    "dataset",
    object_name
)

os.makedirs(
    dataset_path,
    exist_ok=True
)

print("\nSelect Data Collection Method")

print(
    "1. Google + Bing + Baidu Scraper"
)

print(
    "2. Webcam Collection"
)

print(
    "3. Google + Bing + Baidu + Webcam "
    "(Recommended)"
)

choice = input(
    "\nEnter choice (1/2/3): "
).strip()

if choice == "1":

    download_images(
        object_name,
        dataset_path
    )

elif choice == "2":

    collect_from_camera(
        object_name,
        dataset_path
    )

elif choice == "3":

    download_images(
        object_name,
        dataset_path
    )

    collect_from_camera(
        object_name,
        dataset_path
    )

else:

    print("Invalid choice.")
    exit()

clean_dataset(
    dataset_path
)

final_count = len(
    os.listdir(dataset_path)
)

if final_count < 100:

    print(
        "\nWARNING: Less than "
        "100 images collected."
    )

    print(
        "For better accuracy, "
        "collect at least "
        "200-500 images per class."
    )

print("\n==============================")

print(
    "DATASET CREATION COMPLETE"
)

print("==============================")

print(
    f"Object      : {object_name}"
)

print(
    f"Images      : {final_count}"
)

print(
    f"Folder      : {dataset_path}"
)

print("==============================")