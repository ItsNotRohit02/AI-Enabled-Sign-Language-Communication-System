import os
import random
import numpy as np
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor

root_signs_folder = 'DatasetBG'
backgrounds_folder = 'Backgrounds'
output_root_folder = 'Output2'

output_size = (128, 128)
gaussian_noise_ratio = 0.1


def add_gaussian_noise(image, mean=0, stddev=25):
    np_image = np.array(image)
    noise = np.random.normal(mean, stddev, np_image.shape).astype(np.uint8)
    noisy_image = Image.fromarray(np.clip(np_image + noise, 0, 255).astype(np.uint8))
    return noisy_image


background_images = [os.path.join(backgrounds_folder, f) for f in os.listdir(backgrounds_folder) if
                     os.path.isfile(os.path.join(backgrounds_folder, f))]

os.makedirs(output_root_folder, exist_ok=True)


def process_image(sign_path):
    sign_image = Image.open(sign_path).convert('RGBA')
    sign_name = os.path.splitext(os.path.basename(sign_path))[0]
    root = os.path.dirname(sign_path)

    relative_path = os.path.relpath(root, root_signs_folder)
    output_folder = os.path.join(output_root_folder, relative_path)
    os.makedirs(output_folder, exist_ok=True)

    if random.random() < gaussian_noise_ratio:
        background = Image.new('RGBA', output_size, (0, 0, 0, 0))
        noisy_background = add_gaussian_noise(background)
        combined = Image.alpha_composite(noisy_background.convert('RGBA'), ImageOps.fit(sign_image, output_size))
        combined.convert('RGB').save(os.path.join(output_folder, f'{sign_name}_noisy.jpg'), 'JPEG')
    else:
        background_path = random.choice(background_images)
        background = Image.open(background_path).convert('RGBA')
        background = ImageOps.fit(background, output_size)
        combined = Image.alpha_composite(background, ImageOps.fit(sign_image, output_size))
        combined.convert('RGB').save(
            os.path.join(output_folder, f'{sign_name}_on_{os.path.splitext(os.path.basename(background_path))[0]}.jpg'),
            'JPEG')


sign_image_paths = []
for root, _, files in os.walk(root_signs_folder):
    for file in files:
        if file.endswith('.png'):
            sign_image_paths.append(os.path.join(root, file))

with ThreadPoolExecutor() as executor:
    executor.map(process_image, sign_image_paths)

print("Processing complete.")
