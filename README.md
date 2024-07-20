# AI-Enabled-Sign-Language-Communication-System
An AI-enabled system for recognizing Indian Sign Language (ISL) using advanced deep learning models. This project integrates sign detection, classification, text translation, and text-to-speech functionalities.

## Features
- Real-time Sign Detection and Classification: Capture images and detect ISL signs using YOLOv8 and classify them using ViT (Vision Transformer).
- Text Translation: Translate detected sign language into multiple languages including Hindi, Kannada, English, Telugu, Malayalam, and Tamil.
- Text-to-Speech Conversion: Convert translated text into speech using the BarkModel.

### app.py

The `app.py` file is the main application script for the AI Enabled Sign Language System. This script sets up a Streamlit web application that allows users to interact with various functionalities of the system. Here is a detailed explanation of its components and functionalities:

- **Streamlit Setup**: The script configures the Streamlit page, setting the title to "AI Enabled Sign Language System" and adding a page icon.
- **Model Loading**: The YOLOv8 M model for sign detection, ViT model for sign classification, and BarkModel for text-to-speech are loaded. These models are pretrained and loaded onto the appropriate device (CPU or GPU).
- **Sign Classification**: Users can capture an image using their webcam. The captured image is processed to detect signs using YOLOv8. Detected sign images are then classified using the ViT model to predict the corresponding ISL character and its confidence score. The results are displayed alongside the processed images.
- **Text Translation**: The application provides a text input field where users can enter text to be translated. Users can select the target language from a dropdown menu, and upon clicking the translate button, the text is translated using the Deep Translator library. The translated text is then displayed.
- **Text-to-Speech**: Users can enter text they want to convert to speech and select a voice preset. The BarkModel is used to generate speech from the entered text. The generated audio is then played back to the user.

Overall, `app.py` provides an interactive interface for real-time sign detection and classification, text translation, and text-to-speech conversion.

### allsteps.py

The `allsteps.py` script guides users through a step-by-step process for using the AI Enabled Sign Language System. Each step is designed to perform a specific task, ensuring users follow a structured workflow. Here's a breakdown of its components and functionalities:

- **Step 1: Displaying Input Images**: When the user clicks the "Begin" button, the script displays all images from a specified input folder. These images are the ones that will be processed in the subsequent steps.
- **Step 2: Object Detection with YOLO**: In this step, the script performs object detection on the displayed input images using the YOLOv8 model. It annotates the detected signs and crops the detected sign regions from the images. The annotated images and cropped images are displayed to the user.
- **Step 3: Classification with ViT**: The cropped images from the previous step are classified using the ViT model. The script predicts the ISL character for each cropped image and displays the predicted class along with the confidence score. The predictions are saved to a text file.
- **Step 4: Text Translation**: The combined text from the predictions is read from the file. Users can select a language for translation, and the script translates the combined text using the Deep Translator library. The translated text is displayed to the user.
- **Step 5: Text-to-Speech**: The translated text is used as input for text-to-speech conversion. Users select a voice preset and the script generates speech using the BarkModel. The generated audio is played back to the user.

The `allsteps.py` script provides a comprehensive, step-by-step guide for users to follow, ensuring they utilize the system's full capabilities in an orderly manner. Each step is clearly defined, and the user progresses through the stages of image processing, classification, translation, and speech generation.
