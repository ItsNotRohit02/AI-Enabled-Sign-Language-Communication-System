import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def convert_png_to_jpeg(input_output_tuple, quality=85, background_color=(255, 255, 255)):
    input_path, output_path = input_output_tuple
    img = Image.open(input_path)
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, background_color)
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert('RGB')
    img.save(output_path, 'JPEG', quality=quality)


def process_folders(base_folder, output_base_folder, quality=85, max_workers=4, background_color=(255, 255, 255)):
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(base_folder):
            relative_path = os.path.relpath(root, base_folder)
            output_dir = os.path.join(output_base_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            for file in files:
                if file.lower().endswith('.png'):
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.jpg')
                    tasks.append(
                        executor.submit(convert_png_to_jpeg, (input_file, output_file), quality, background_color))

        for future in as_completed(tasks):
            future.result()


base_folder = "Dataset"
output_base_folder = "Output2"
process_folders(base_folder, output_base_folder, quality=85, max_workers=8, background_color=(0, 0, 0))
