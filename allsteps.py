import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification, BarkModel, AutoProcessor
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import scipy.io.wavfile
import tempfile
import time

st.set_page_config(page_title="AI Enabled Sign Language System", page_icon="💻")
yolo_model_path = 'models/ISL-YOLOv8mBoundingBox.pt'
yolo_model = YOLO(yolo_model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
vit_model.load_state_dict(torch.load("models/ISL-ViTImageClassification.pth", map_location=device))
vit_model.to(device)
vit_model.eval()

bark_model = BarkModel.from_pretrained("suno/bark-small")
bark_model.to(device)
bark_processor = AutoProcessor.from_pretrained("suno/bark")

class_names = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = vit_model(image).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    return predicted_class, confidence_score


def detect_objects(image):
    results = yolo_model(image)
    annotated_image = image.copy()
    cropped_images = []

    for result in results[0].boxes:
        if result.conf.item() >= 0.50:
            x1, y1, x2, y2 = map(int, result.xyxy.tolist()[0])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

    return annotated_image, cropped_images


def main():
    st.title('AI Enabled Sign Language System')

    if 'start' not in st.session_state:
        st.session_state.start = 1
        st.session_state.step = 0
        st.session_state.input_images = []
        st.session_state.cropped_images = []
        st.session_state.input_folder = 'images'
        st.session_state.output_folder = 'cropped_images'

    if st.button('Begin'):
        st.session_state.step = 1

    if (st.session_state.step == 1):
        st.header("Step 1: Displaying Input Images")
        os.makedirs(st.session_state.output_folder, exist_ok=True)

        st.session_state.input_images = [os.path.join(st.session_state.input_folder, img) for img in
                                         os.listdir(st.session_state.input_folder) if
                                         img.endswith('.png') or img.endswith('.jpg')]
        for img_path in st.session_state.input_images:
            img = Image.open(img_path)
            st.image(img, caption=os.path.basename(img_path), use_column_width=True)
        if st.button('Start Processing'):
            st.session_state.step = 2

    if (st.session_state.step == 2):
        st.header("Step 2: Object Detection with YOLO")

        annotated_images = []
        cropped_images = []
        for img_path in st.session_state.input_images:
            image = cv2.imread(img_path)
            image = cv2.flip(image, 1)
            annotated_image, crops = detect_objects(image)
            annotated_images.append(annotated_image)
            cropped_images.extend(crops)
            for i, crop in enumerate(crops):
                crop_path = os.path.join(st.session_state.output_folder,
                                         f"cropped_{os.path.basename(img_path).split('.')[0]}_{i}.png")
                cv2.imwrite(crop_path, crop)

        for ann_img in annotated_images:
            ann_img_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
            st.image(ann_img_rgb, caption='Annotated Image', use_column_width=True)

        st.session_state.cropped_images = cropped_images

        st.success(f"All Input Images have been Processed")

        if st.button('Classify Images'):
            st.session_state.step = 3

    if (st.session_state.step == 3):
        st.header("Step 3: Classification with ViT")

        predictions = []

        for crop in st.session_state.cropped_images:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            predicted_class, confidence_score = predict_image(crop_pil)
            predictions.append(predicted_class)
            st.image(crop_pil, caption=f'Predicted: {predicted_class} ({confidence_score:.2f}%)', use_column_width=True)

        with open('predictions.txt', 'w') as f:
            f.write(''.join(predictions))

        with open('predictions.txt', 'r') as f:
            combined_text = f.read()
        st.write(f'Combined Text: {combined_text}')

        st.success(f"All Images have been Classified")

        if st.button('Text Translation'):
            st.session_state.step = 4

    if (st.session_state.step == 4):
        st.header("Step 4: Text Translation")
        with open('predictions.txt', 'r') as f:
            combined_text = f.read()

        st.write(f'Combined Text: {combined_text}')

        lang = st.selectbox(
            'Which language do you want to translate it to?',
            ('', 'Hindi', 'Kannada', 'English', 'Telugu', 'Malayalam', 'Tamil'))
        if lang == '':
            lang = None

        trns = st.button('Translate')

        if lang and trns:
            if lang == 'Hindi':
                option = 'hi'
            elif lang == 'Kannada':
                option = 'kn'
            elif lang == 'English':
                option = 'en'
            elif lang == 'Telugu':
                option = 'te'
            elif lang == 'Malayalam':
                option = 'ml'
            elif lang == 'Tamil':
                option = 'ta'

            translated_text = GoogleTranslator(source='auto', target=option).translate(combined_text)
            st.write(f'Translated Text: {translated_text}')
            st.success(f"Text has been successfully translated to {lang}")

        if st.button('Next Step'):
            st.session_state.step = 5

    if (st.session_state.step == 5):

        st.header("Step 5: Text to Speech")
        with open('predictions.txt', 'r') as f:
            combined_text = f.read()

        st.write(f'Combined Text: {combined_text}')

        voice_preset_option = st.selectbox(
            'Select a voice preset',
            ('', 'English', 'Hindi')
        )
        if voice_preset_option == '':
            voice_preset_option = None

        generate_speech = st.button("Generate Speech")

        if generate_speech and voice_preset_option:
            if voice_preset_option == 'English':
                voice_preset_option = 'v2/en_speaker_6'
            elif voice_preset_option == 'Hindi':
                voice_preset_option = 'v2/hi_speaker_2'

            inputs = bark_processor(combined_text, voice_preset=voice_preset_option)
            speech_output = bark_model.generate(**inputs.to(device))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                scipy.io.wavfile.write(tmpfile.name, rate=bark_model.generation_config.sample_rate,
                                       data=speech_output[0].cpu().numpy())
                st.audio(tmpfile.name, format="audio/wav")
                st.success(f"Audio Generated")


if __name__ == "__main__":
    main()
