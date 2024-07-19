import os
import random
import shutil


def split_train_test(base_folder, output_train_folder, output_test_folder, split_ratio=0.8):
    for root, _, files in os.walk(base_folder):
        relative_path = os.path.relpath(root, base_folder)
        train_dir = os.path.join(output_train_folder, relative_path)
        test_dir = os.path.join(output_test_folder, relative_path)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)

        for i, image in enumerate(images):
            src_path = os.path.join(root, image)
            if i < split_index:
                dst_path = os.path.join(train_dir, image)
            else:
                dst_path = os.path.join(test_dir, image)
            shutil.copy(src_path, dst_path)


base_folder = "Dataset"
output_train_folder = "DatasetBG/train"
output_test_folder = "DatasetBG/test"
split_train_test(base_folder, output_train_folder, output_test_folder, split_ratio=0.85)
