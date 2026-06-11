import os
import imagehash
from PIL import Image
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler


MAX_IMAGES_GOOGLE = 500
MAX_IMAGES_BING = 500


object_name = input("Enter object name: ").strip()

dataset_path = os.path.join("dataset", object_name)
os.makedirs(dataset_path, exist_ok=True)

print(f"\nCreating dataset for: {object_name}")
print("Downloading images...\n")


google = GoogleImageCrawler(
    storage={"root_dir": dataset_path}
)

google.crawl(
    keyword=object_name,
    max_num=MAX_IMAGES_GOOGLE
)


bing = BingImageCrawler(
    storage={"root_dir": dataset_path}
)

bing.crawl(
    keyword=object_name,
    max_num=MAX_IMAGES_BING
)

print("Download complete.")


print("Checking corrupted files...")

for filename in os.listdir(dataset_path):

    filepath = os.path.join(dataset_path, filename)

    try:
        img = Image.open(filepath)
        img.verify()

    except Exception:
        try:
            os.remove(filepath)
            print("Removed corrupted:", filename)
        except Exception as e:
            print(e)


print("Removing duplicates...")

hashes = {}
duplicates = 0

for filename in os.listdir(dataset_path):

    filepath = os.path.join(dataset_path, filename)

    try:
        image = Image.open(filepath)

        img_hash = str(
            imagehash.phash(image)
        )

        if img_hash in hashes:
            os.remove(filepath)
            duplicates += 1
        else:
            hashes[img_hash] = filename

    except:
        pass

print(f"Removed {duplicates} duplicate images.")

final_count = len(os.listdir(dataset_path))

print("\n==============================")
print("DATASET CREATION COMPLETE")
print("==============================")
print(f"Object      : {object_name}")
print(f"Images      : {final_count}")
print(f"Folder      : {dataset_path}")
print("==============================")