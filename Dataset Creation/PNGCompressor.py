import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def compress_png(input_output_tuple, quality=75):
    input_path, output_path = input_output_tuple
    img = Image.open(input_path)
    img.save(output_path, optimize=True, quality=quality)


def process_folders(base_folder, output_base_folder, quality=75, max_workers=4):
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(base_folder):
            relative_path = os.path.relpath(root, base_folder)
            output_dir = os.path.join(output_base_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            for file in files:
                if file.lower().endswith('.png'):
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_dir, file)
                    tasks.append(executor.submit(compress_png, (input_file, output_file), quality))

        for future in as_completed(tasks):
            future.result()


base_folder = "Dataset"
output_base_folder = "Output"
process_folders(base_folder, output_base_folder, quality=75, max_workers=8)
