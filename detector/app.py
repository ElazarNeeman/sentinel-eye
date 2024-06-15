import time
import cv2

from detection_aggragate import DetectionAggregate
from detection_alarm import DetectionAlarm
from single_frame_detector import Detector
from telegram import client
from video import VideoStream

video_stream = VideoStream(stream_id=0)  # stream_id = 0 is for primary camera
detector = Detector()
detection_aggregate = DetectionAggregate()
alarm = DetectionAlarm(client)
video_stream.start()

# processing frames in input stream
num_frames_processed = 0

start = time.time()
old_time = start


def ten_sec_passed(old_epoch):
    return frame_time - old_epoch >= 10


while not video_stream.stopped:

    frame = video_stream.read()
    detector.process(frame)
    frame_time = detection_aggregate.add_detector_frame(detector)

    num_frames_processed += 1

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if ten_sec_passed(old_time):
        old_time = frame_time
        if detection_aggregate.has_detections():
            print(f"{time.ctime(frame_time)}: detections: {detection_aggregate.get_data()}")
            try:
                alarm.add_detection(detection_aggregate)
            except Exception as e:
                print(e)

        detection_aggregate = DetectionAggregate()

    if key == ord('q'):
        break

end = time.time()
video_stream.stop()  # stop the webcam stream
alarm.stop()  # stop the alarm thread

# printing time elapsed and fps
elapsed = end - start
fps = num_frames_processed / elapsed
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows
cv2.destroyAllWindows()
