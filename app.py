import streamlit as st
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


st.title('AI Enabled Sign Language System')

st.header("Classification of Sign")

img_file_buffer = st.camera_input("**Capture an image**")

if img_file_buffer is not None:
    input_image = Image.open(img_file_buffer).convert('RGB')
    input_image_cv2 = np.array(input_image)
    input_image_cv2 = cv2.cvtColor(input_image_cv2, cv2.COLOR_RGB2BGR)

    input_image_cv2 = cv2.flip(input_image_cv2, 1)

    annotated_image, cropped_images = detect_objects(input_image_cv2)

    col1, col2 = st.columns(2)

    with col1:
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.write('**Annotated Image**')
        st.image(annotated_image_rgb, use_column_width=True)

    for i, cropped_image in enumerate(cropped_images):
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_image_rgb)
        predicted_class, confidence_score = predict_image(cropped_pil)

        with col2:
            st.write(f'**Cropped Image** {i + 1}')
            st.image(cropped_pil, use_column_width=True)

        st.success(f'Predicted Sign is {predicted_class} with Confidence of {confidence_score:.2f}%')

st.write('---')
st.header('Text Translation')

enterorg = st.text_input("Enter the text to be translated")
lang = st.selectbox(
    'Which language do you want to translate it to?',
    ('', 'Hindi', 'Kannada', 'English', 'Telugu', 'Malayalam', 'Tamil'))
trns = st.button('Translate')

if (lang == ''):
    option = None

if trns and lang:
    if (lang == 'Hindi'):
        option = 'hi'
    elif (lang == 'Kannada'):
        option = 'kn'
    elif (lang == 'English'):
        option = 'en'
    elif (lang == 'Telugu'):
        option = 'te'
    elif (lang == 'Malayalam'):
        option = 'ml'
    elif (lang == 'Tamil'):
        option = 'ta'

    translated_text = GoogleTranslator(source='auto', target=option).translate(enterorg)
    time.sleep(2)
    st.write(f'Translated Text: {translated_text}')
    st.success(f"Text has been successfully translated to {lang}")

st.write('---')
st.header("Text to Speech")

text_prompt = st.text_area("Enter the text you want to convert to speech:")
voice_preset_option = st.selectbox(
    'Select a voice preset',
    ('', 'English', 'Hindi')
)
if (voice_preset_option == ''):
    voice_preset_option = None

generate_speech = st.button("Generate Speech")

if generate_speech and text_prompt and voice_preset_option:
    if (voice_preset_option == 'English'):
        voice_preset_option = 'v2/en_speaker_6'
    elif (option == 'Hindi'):
        voice_preset_option = 'v2/hi_speaker_2'

    inputs = bark_processor(text_prompt, voice_preset=voice_preset_option)
    speech_output = bark_model.generate(**inputs.to(device))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, rate=bark_model.generation_config.sample_rate,
                               data=speech_output[0].cpu().numpy())
        st.audio(tmpfile.name, format="audio/wav")
        st.success(f"Audio Generated")
