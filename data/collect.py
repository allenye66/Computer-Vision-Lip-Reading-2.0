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


TOTAL_FRAMES = 22
VALID_WORD_THRESHOLD = 5
NOT_TALKING_THRESHOLD = 10
PAST_BUFFER_SIZE = 4
# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("../model/face_weights.dat")

# read the image
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 60)
all_words = []
curr_word_frames = []
not_talking_counter = 0


LIP_WIDTH = 112
LIP_HEIGHT = 80

data_count = 1
label = input("What word you like to collect data for? ")
labels = []

past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)

ending_buffer_size = 5




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




        print(len(curr_word_frames))

        if lip_distance > 45: # person is talking
            cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            curr_word_frames += [lip_frame.tolist()]
            not_talking_counter = 0

        else:
            cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            not_talking_counter += 1
            
            # a valid word finished and has all needed ending buffer frames
            # we do len(curr_word_frames) + PAST_BUFFER_SIZE since we add past frames after this step (not included yet)
            if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES: 
                data_count += 1
                curr_word_frames = list(past_word_frames) + curr_word_frames
                print(f"adding {label.upper()} shape", lip_frame.shape, "count is", data_count, "frames is", len(curr_word_frames))

                all_words.append(curr_word_frames)
                labels.append(label)
                curr_word_frames = []
                not_talking_counter = 0

            # curr word frames not fully done yet, add ending buffer frames
            elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:
                #print("adding ending buffer frames the len(curr_word_frames) is", (len(curr_word_frames)))
                curr_word_frames += [lip_frame.tolist()]
                not_talking_counter = 0
            # too little frames, discard the data
            elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                #print("bad recording, resetting curr word frames")
                curr_word_frames = []

            past_word_frames+= [lip_frame.tolist()]

            #circular frame buffer
            if len(past_word_frames) > PAST_BUFFER_SIZE:
                past_word_frames.pop(0)

    cv2.imshow(winname="Mouth", mat=frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

#not needed for new version where we have a set amount of frames
def process_frames(all_words, labels):
    # Get the median length of all sublists
    median_length = statistics.median([len(sublist) for sublist in all_words])
    median_length = int(median_length)
    # Remove sublists shorter than the median length
    print("Removing sublists shorter than the median length")
    indices_to_keep = [i for i, sublist in enumerate(all_words) if (len(sublist) >= median_length and  len(sublist) <= median_length + 2)]
    all_words = [all_words[i] for i in indices_to_keep]
    labels = [labels[i] for i in indices_to_keep]

    # Truncate all remaining sublists to the median length
    all_words = [sublist[:median_length] for sublist in all_words]

    return all_words, labels


all_words, labels = process_frames(all_words, labels)


def saveAllWords(all_words):

    print("saving words into dir!")
    """
    Creates a folder and subfolders for each set of curr_word_frames inside all_words, and saves the
    frames as images inside their corresponding subfolders.
    
    Parameters:
        all_words (list): A 3D list containing the frames for each word spoken.
    """
    output_dir = "../outputs"
    next_dir_number = 1
    for i, word_frames in enumerate(all_words):

        label = labels[i]

        word_folder = os.path.join(output_dir, label + "_" + f"{next_dir_number}")
        while os.path.exists(word_folder):
            next_dir_number += 1
            word_folder = os.path.join(output_dir, label + "_" + f"{next_dir_number}")
        os.makedirs(word_folder)

        txt_path = os.path.join(word_folder, "data.txt")

        with open(txt_path, "w") as f:
            f.write(json.dumps(word_frames))

        images = []

        for j, img_data in enumerate(word_frames):
            img = Image.new('RGB', (len(img_data[0]), len(img_data)))
            pixels = img.load()
            for y in range(len(img_data)):
                for x in range(len(img_data[y])):
                    pixels[x, y] = tuple(img_data[y][x])
            img_path = os.path.join(word_folder, f"{j}.png")
            img.save(img_path)
            images.append(imageio.imread(img_path))
        print("The length of this subfolder:", len(images))
        video_path = os.path.join(word_folder, "video.mp4")
        imageio.mimsave(video_path, images, fps=int(cap.get(cv2.CAP_PROP_FPS)))
        next_dir_number += 1

saveAllWords(all_words)
# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()