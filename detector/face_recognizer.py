import time

import cv2
import json

from deepface import DeepFace

from detection_aggragate import DetectionAggregate
from detection_alarm import DetectionAlarm
from telegram import client

if __name__ == "__main__":

    def ten_sec_passed(old_epoch):
        return frame_time - old_epoch >= 10


    # Create LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Load the trained model
    recognizer.read('trainer.yml')
    print(recognizer)
    # Path to the Haar cascade file for face detection
    face_cascade_Path = "haarcascade_frontalface_default.xml"

    # Create a face cascade classifier
    faceCascade = cv2.CascadeClassifier(face_cascade_Path)

    # Font for displaying text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize user IDs and associated names
    id = 0
    # Don't forget to add names associated with user IDs
    names = ['None']
    with open('names.json', 'r') as fs:
        names = json.load(fs)
        names = list(names.values())

    # Video Capture from the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture("rtsp://10.100.102.61:8554/stream")

    # cam = cv2.VideoCapture("rtsp://10.100.102.62:8554/substream")
    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    # Minimum width and height for the window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    detection_aggregate = DetectionAggregate()
    alarm = DetectionAlarm(client)

    old_time = time.time()
    skip_frames = 1
    cnt = 1

    while True:
        # Read a frame from the camera
        cnt += 1
        ret, img = cam.read()
        frame_time = time.time()

        cv2.imshow('camera', img)

        if (cnt % skip_frames) != 0:
            continue

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(e)
            continue

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recognize the face using the trained model
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            detected_face = img[y:y + h, x:x + w]

            confidence_numeric = round(confidence)

            demographies = DeepFace.analyze(
                img_path=detected_face,
                actions=("emotion",),
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            # print(demographies)

            # Proba greater than 51
            if confidence > 51:
                try:
                    # Recognized face
                    name = names[id]
                    confidence = "  {0}%".format(round(confidence))


                except IndexError as e:
                    name = "Who are you?"
                    confidence = "N/A"
            else:
                # Unknown face
                name = "Who are you?"
                confidence = "N/A"

            dominant_emotion = demographies[0]['dominant_emotion']
            # Display the recognized name and confidence level on the image
            cv2.putText(img, f"{name}, ({dominant_emotion})", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            # add detection
            detection_aggregate.add_face_detection(name, confidence_numeric, dominant_emotion, frame_time,
                                                   detected_face)

        # Display the image with rectangles around faces
        detection_aggregate.add_frame()
        cv2.imshow('camera', img)

        # skip_frames to allow time for processing
        if detection_aggregate.has_detections():
            skip_frames = 5
        else:
            skip_frames = 1

        if ten_sec_passed(old_time):
            old_time = frame_time
            if detection_aggregate.has_detections():
                print(f"{time.ctime(frame_time)}: detections: {detection_aggregate.get_data()}")

                try:
                    alarm.add_detection(detection_aggregate)
                except Exception as e:
                    print(e)

            detection_aggregate = DetectionAggregate()

            # Press Escape to exit the webcam / program
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    print("\n [INFO] Exiting Program.")
    # Release the camera
    cam.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
