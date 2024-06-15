import time

import cv2
from telethon import TelegramClient
import queue
import threading
from detection_aggragate import DetectionAggregate


class DetectionAlarm:

    def __init__(self, client: TelegramClient):
        self.agg = DetectionAggregate()
        self.client = client
        self.alarm_time = {}
        self.alarm_queue = queue.Queue()
        self.alarm_thread = threading.Thread(target=self.alarm_worker)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()

    def alarm_worker(self):
        while True:
            alarm_data, face = self.alarm_queue.get()
            if alarm_data is None:
                break
            self.process_alarm(alarm_data, face)
            self.alarm_queue.task_done()

    def process_alarm(self, alarm_data, face):
        name = alarm_data['name']
        print(f"{time.ctime()} Raising alarm for {alarm_data}")
        alarm_time = time.time()
        file_name = f'./alarms/Users-{round(alarm_time)}.jpg'
        cv2.imwrite(file_name, face)

        if self.client:
            self.client.send_file('+972545664107', file_name,
                                  caption=f'Person {name} detected at {time.ctime()}, info: {alarm_data}')

        self.alarm_time[name] = alarm_time

    def raise_alarm(self, alarm_data, face):
        self.alarm_queue.put((alarm_data, face))

    def stop(self):
        self.alarm_queue.put((None, None))
        self.alarm_thread.join()

    def add_detection(self, da: DetectionAggregate, min_frames=5, last_seen_minutes=15):
        detections = da.detections

        for name in detections:
            if detections[name]['count'] < min_frames:
                continue

            if name not in self.agg.detections:
                self.raise_alarm({'name': name, **detections[name], }, da.faces[name])
                continue

            name_status = self.agg.detections[name]
            if time.time() > name_status['last_frame_time_epoch'] + last_seen_minutes * 60:
                self.raise_alarm({'name': name, **detections[name], }, da.faces[name])

        self.agg = DetectionAggregate.combine(self.agg, da)
