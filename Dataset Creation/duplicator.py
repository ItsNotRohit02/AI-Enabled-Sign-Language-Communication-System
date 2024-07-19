import os
import shutil


def duplicate_images_in_folder(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_count = len(images)

    if image_count >= 1000:
        return

    index = 0
    while image_count < 1000:
        src_image = os.path.join(folder_path, images[index])
        dst_image = os.path.join(folder_path, f"{image_count + 1:04d}_{os.path.basename(images[index])}")
        shutil.copy(src_image, dst_image)
        image_count += 1
        index = (index + 1) % len(images)


def process_folders(base_folder):
    for root, dirs, files in os.walk(base_folder):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            duplicate_images_in_folder(folder_path)


base_folder = "Dataset"
process_folders(base_folder)
