import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.initializers import GlorotUniform, Zeros

# Initialize the model
model = Sequential(name='sequential')
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1),
                 kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), name='conv2d'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d'))
model.add(Dropout(0.4, name='dropout'))

model.add(Conv2D(256, (3, 3), activation='relu',
                 kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), name='conv2d_1'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_1'))
model.add(Dropout(0.4, name='dropout_1'))

model.add(Conv2D(512, (3, 3), activation='relu',
                 kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), name='conv2d_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_2'))
model.add(Dropout(0.4, name='dropout_2'))

model.add(Conv2D(512, (3, 3), activation='relu',
                 kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), name='conv2d_3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_3'))
model.add(Dropout(0.4, name='dropout_3'))

model.add(Flatten(name='flatten'))
model.add(Dense(512, activation='relu',
                kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), name='dense'))
model.add(Dropout(0.4, name='dropout_4'))

model.add(Dense(256, activation='relu',
                kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), name='dense_1'))
model.add(Dropout(0.3, name='dropout_5'))

model.add(Dense(7, activation='softmax',
                kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), name='dense_2'))

# Load the pre-trained weights
model.load_weights("facialemotionmodel.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start video capture
webcam = cv2.VideoCapture(0)  # Change index to 0 for the default camera
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        cv2.putText(im, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow("Output", im)

    if cv2.waitKey(1) & 0xFF == 27:  # If 'ESC' key is pressed, break the loop
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()