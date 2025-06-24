# 😃 Facial Emotion Recognition using CNN and OpenCV

This project detects human facial emotions in real time using a webcam. It uses a Convolutional Neural Network (CNN) trained on 48x48 grayscale face images and predicts one of 7 emotions.

## 📂 Files Included

- `facialemotionmodel.h5` – Trained CNN model weights  
- `facialemotionmodel.json` – Model architecture in JSON format  
- `main.py` – Python script to load model and run emotion detection  
- `new.py` – Alternate script with full model definition (can retrain/load weights)

## 📸 Emotions Detected

- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😊 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise

## 💻 How it Works

1. The webcam captures a live video feed.
2. Faces are detected using OpenCV’s Haar Cascade.
3. Each face is resized to 48x48 and passed to the CNN.
4. The model predicts the most probable emotion.
5. The emotion is displayed on the screen near the detected face.

## ▶️ How to Run

1. Install required libraries:
   ```bash
   pip install opencv-python tensorflow numpy
2. Run the script:
bash
Copy
Edit
python main.py
(or new.py, which includes model architecture too)

3. Press Esc to exit the webcam window.

🧠 Model Architecture

4. Conv2D layers with ReLU activation

   MaxPooling2D after each conv layer

   Dropout layers to reduce overfitting

   Dense layers: 512 → 256 → 7 (softmax)

🧰 Libraries Used

OpenCV

TensorFlow / Keras

NumPy

📌 Notes

Make sure your webcam is enabled.

Works on grayscale face images of 48x48 pixels.

You can extend this for video analysis, mobile apps, or integration with GUI.


🙋‍♀️ Created By

Mohanapriya – Final Year IT Student at Annamalai University


