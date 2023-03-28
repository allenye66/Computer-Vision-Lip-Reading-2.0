import os
import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import csv
from collections import deque
import tensorflow as tf
# Define the input shape
input_shape = (17, 80, 112, 3)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.load_weights('../model/model_weights.h5', by_name=True)

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("../model/face_weights.dat")

# read the image
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 60)
curr_word_frames = []
not_talking_counter = 0


LIP_WIDTH = 112
LIP_HEIGHT = 80


first_word = True
labels = []

past_buffer_size = 4
past_word_frames = deque(maxlen=past_buffer_size)

ending_buffer_size = 5

predicted_word_label = None
draw_prediction = False
while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)
    
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Calculate the distance between the upper and lower lip landmarks
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])



        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        # Add padding if necessary to get a 76x110 frame
        width_diff = LIP_WIDTH - (lip_right - lip_left)
        height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top

        # Ensure that the padding doesn't extend beyond the original frame
        pad_left = min(pad_left, lip_left)
        pad_right = min(pad_right, frame.shape[1] - lip_right)
        pad_top = min(pad_top, lip_top)
        pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

        # Create padded lip region
        lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

        
        lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
        # Apply contrast stretching to the L channel of the LAB image
        l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
        l_channel_eq = clahe.apply(l_channel)

        # Merge the equalized L channel with the original A and B channels
        lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
        lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
        lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
        lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
        kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

        # Apply the kernel to the input image
        lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
        lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
        lip_frame = lip_frame_eq
        
        
        # Draw a circle around the mouth
        for n in range(48, 61):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

        if lip_distance > 45: # person is talking
            cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            curr_word_frames += [lip_frame.tolist()]
        
            not_talking_counter = 0
            draw_prediction = False
        else:
            cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            not_talking_counter += 1
            if not_talking_counter >= 10 and len(curr_word_frames) > 3: # word finished
                curr_word_frames = list(past_word_frames) + curr_word_frames

                curr_data = np.array([curr_word_frames[:input_shape[0]]])
  
                print(curr_data.shape)
                if (curr_data.shape[1]) == input_shape[0]:

                    print("PREDICTING NOW!")
                    prediction = model.predict(curr_data)
                    
                    if prediction < 0.5:
                        predicted_word_label = "COOP"
                    else:
                        predicted_word_label = "CAT"
                    print("FINISHED!", predicted_word_label)
                    # Draw the predicted word on the screen
                    draw_prediction = True

                curr_word_frames = []
                not_talking_counter = 0
            #'''(input_shape[0]- len(curr_word_frames) + past_buffer_size - 1) '''
            elif not_talking_counter < ending_buffer_size and len(curr_word_frames) > 3: #add ending buffer frames, make it dynamic
                curr_word_frames += [lip_frame.tolist()]
            past_word_frames+= [lip_frame.tolist()]
            if len(past_word_frames) > past_buffer_size:
                past_word_frames.pop(0)

    if(draw_prediction):
        cv2.putText(frame, predicted_word_label, (50 ,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.imshow(winname="Mouth", mat=frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break


cap.release()

# Close all windows
cv2.destroyAllWindows()