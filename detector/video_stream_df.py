# importing required libraries
import os

import cv2
import time
from threading import Thread  # library for implementing multi-threaded processing

import numpy as np
from deepface import DeepFace


# defining a helper class for implementing multi-threaded processing
class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for primary camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background while the program is executing

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()

        # method for reading next frame

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def stop(self):
        self.stopped = True

    # initializing and starting multi-threaded webcam capture input stream


# we are not going to bother with objects less than 50% probability
THRESHOLD = 0.4
# the lower the value: the fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.3
SSD_INPUT_SIZE = 320


def get_name(_res):
    if len(_res) > 0 and len(_res[0]['identity']) > 0:
        identity_path = _res[0]['identity'][0]
        _, identity_file = os.path.split(identity_path)

        # print(identity_file)
        name = identity_file.split('-')[-2]
        return name

    return None


# read the class labels
def construct_class_names(file_name='class_names'):
    with open(file_name, 'rt') as file:
        names = file.read().rstrip('\n').split('\n')

    return names


def show_detected_objects(img, boxes_to_keep, all_bounding_boxes, object_names, class_ids):
    for index in boxes_to_keep:
        box = all_bounding_boxes[index]
        x, y, w, h = box[0], box[1], box[2], box[3]
        obj_class_name = object_names[class_ids[index] - 1]
        print(obj_class_name)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)
        cv2.putText(img, obj_class_name.upper(), (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 2)

        if obj_class_name == "person":
            if cnt % skip_frames == 0:
                show_detected_faces(img, (x, y, w, h))


def show_detected_faces(img, person_position):
    try:
        (px, py, pw, ph) = person_position
        gray = cv2.cvtColor(img[py:py + ph, px:px + pw], cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(e)
        return

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        x = x + px
        y = y + py
        detected_face = img[y:y + h, x:x + w]

        res = DeepFace.find(img_path=detected_face, db_path="im_db_judges1", align=False,
                            enforce_detection=False, silent=True)
        name = get_name(res)

        demographies = DeepFace.analyze(
            img_path=detected_face,
            actions=("emotion",),
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )

        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        confidence = ""

        if name is None:
            name = "Who are you?"

        dominant_emotion = demographies[0]['dominant_emotion']
        # Display the recognized name and confidence level on the image
        cv2.putText(img, f"{name}, ({dominant_emotion})", (x + 5, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 255, 255), 2)
        print(name, " ", dominant_emotion)
        cv2.putText(img, confidence, (x + 5, y + h - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1)
        cv2.imshow(f'Face {name}', detected_face)
        # cv2.imshow(f'Gray Face {name}', gray_face)
        # add detection


class_names = construct_class_names()

webcam_stream = WebcamStream(stream_id=0)  # stream_id = 0 is for primary camera
webcam_stream.start()

neural_network = cv2.dnn_DetectionModel('ssd_weights.pb', 'ssd_mobilenet_coco_cfg.pbtxt')
# define whether we run the algorithm with CPU or with GPU
# WE ARE GOING TO USE CPU !!!
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
neural_network.setInputSize(SSD_INPUT_SIZE, SSD_INPUT_SIZE)
neural_network.setInputScale(1.0 / 127.5)
neural_network.setInputMean((127.5, 127.5, 127.5))
neural_network.setInputSwapRB(True)

# Path to the Haar cascade file for face detection
face_cascade_Path = "haarcascade_frontalface_default.xml"

# Create a face cascade classifier
faceCascade = cv2.CascadeClassifier(face_cascade_Path)

# Minimum width and height for the window size to be recognized as a face
minW = 0.1 * webcam_stream.vcap.get(3)
minH = 0.1 * webcam_stream.vcap.get(4)

cnt = 0
persons = 0
skip_frames = 1

# processing frames in input stream
num_frames_processed = 0
start = time.time()
while True:
    if webcam_stream.stopped is True:
        break
    else:
        frame = webcam_stream.read()

        # adding a delay for simulating time taken for processing a frame

    class_label_ids, confidences, bbox = neural_network.detect(frame)
    bbox = list(bbox)
    confidences = np.array(confidences).reshape(1, -1).tolist()[0]

    # these are the indexes of the bounding boxes we have to keep
    box_to_keep = cv2.dnn.NMSBoxes(bbox, confidences, THRESHOLD, SUPPRESSION_THRESHOLD)
    persons = 0
    show_detected_objects(frame, box_to_keep, bbox, class_names, class_label_ids)

    num_frames_processed += 1

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

end = time.time()
webcam_stream.stop()  # stop the webcam stream

# printing time elapsed and fps
elapsed = end - start
fps = num_frames_processed / elapsed
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows
cv2.destroyAllWindows()
