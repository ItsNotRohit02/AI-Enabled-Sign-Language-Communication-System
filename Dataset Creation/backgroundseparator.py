import cv2
import numpy as np
import os


def process_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)

    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv_image, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv_image, lower_skin2, upper_skin2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    b, g, r = cv2.split(image)
    a = mask
    result = cv2.merge((b, g, r, a))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)


def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.png')

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                process_image(input_path, output_path)


input_directory = 'Dataset/2'
output_directory = 'Output/2'

process_directory(input_directory, output_directory)
