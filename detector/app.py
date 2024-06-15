import time
import cv2
from single_frame_detector import Detector
from video import VideoStream

video_stream = VideoStream(stream_id=0)  # stream_id = 0 is for primary camera
detector = Detector()
video_stream.start()

# processing frames in input stream
num_frames_processed = 0

start = time.time()

while not video_stream.stopped:

    frame = video_stream.read()
    detector.process(frame)
    num_frames_processed += 1

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

end = time.time()
video_stream.stop()  # stop the webcam stream

# printing time elapsed and fps
elapsed = end - start
fps = num_frames_processed / elapsed
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows
cv2.destroyAllWindows()
