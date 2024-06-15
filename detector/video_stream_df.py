import time

import cv2
from deepface import DeepFace

from identity import get_name
from ssd import SingleShotDetector
from video import VideoStream
from viola_jones_face_detector import ViolaJonesFaceDetector


def process(img):
    for x, y, w, h, obj_class_name in ssd.get_detected_objects(img):
        print(obj_class_name)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)
        cv2.putText(img, obj_class_name.upper(), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 2)

        if obj_class_name == "person":
            show_detected_faces(img, (x, y, w, h))


def show_detected_faces(img, person_position):
    for (x, y, w, h, detected_face) in vj.detect_faces(img, person_position):

        res = DeepFace.find(img_path=detected_face, db_path="im_db_team", align=False,
                            detector_backend="skip",
                            enforce_detection=False, silent=True)

        name = get_name(res)

        face_emotions = DeepFace.analyze(img_path=detected_face, actions=("emotion",),
                                         detector_backend="skip",
                                         enforce_detection=False, silent=True)

        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if name is None:
            name = "Who are you?"

        dominant_emotion = face_emotions[0]['dominant_emotion']

        # Display the recognized name and confidence level on the image
        cv2.putText(img, f"{name}, ({dominant_emotion})", (x + 5, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 255, 255), 2)

        print(name, " ", dominant_emotion)
        cv2.imshow(f'Face {name}', detected_face)


webcam_stream = VideoStream(stream_id=0)  # stream_id = 0 is for primary camera
webcam_stream.start()

ssd = SingleShotDetector()
vj = ViolaJonesFaceDetector(webcam_stream)

# processing frames in input stream
num_frames_processed = 0

start = time.time()
while True:
    if webcam_stream.stopped is True:
        break
    else:
        frame = webcam_stream.read()

    process(frame)
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
