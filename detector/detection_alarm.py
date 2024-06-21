import asyncio
import threading
import time

import cv2
from telethon import TelegramClient, events

from detection_aggragate import DetectionAggregate
from telegram import TELEGRAM_CHANNEL_ID, TELEGRAM_API_ID, TELEGRAM_API_HASH


class DetectionAlarm:

    def __init__(self):
        self.agg = DetectionAggregate()
        self.client = None
        self.alarm_time = {}
        self.alarm_queue = asyncio.Queue()
        self.alarm_thread = threading.Thread(target=self.__alarm_thread, args=())
        self.alarm_thread.daemon = True
        self.files = {}

    def __alarm_thread(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.client = TelegramClient('new_session_name', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        self.client.start()

        @self.client.on(events.NewMessage)
        async def on_message(event):
            print("!!! got ", event.raw_text)
            if 'hello' in event.raw_text:
                await event.reply('hi!')
            if 'who' in event.raw_text:
                print('got who')
                await self.__reply_who(event)
            if 'reset' in event.raw_text:
                await self.__reset_detections(event)

        self.client.loop.run_until_complete(self.__alarm_worker())
        self.client.loop.close()

    async def start(self):
        self.alarm_thread.start()

    async def __alarm_worker(self):
        print("Alarm worker started")

        welcome_message = ('Hello, a-eye! I am ready to raise alarms. send me "who" to get people in house, '
                           'and "reset" to reset detections')

        print("telegram channel:", TELEGRAM_CHANNEL_ID)
        await self.client.send_message(TELEGRAM_CHANNEL_ID, welcome_message)

        while True:
            alarm_data, face = await self.alarm_queue.get()
            if alarm_data is None:
                break
            await self.process_alarm(alarm_data, face)
            self.alarm_queue.task_done()

    async def process_alarm(self, alarm_data, face):
        name = alarm_data['name']
        print(f"{time.ctime()} Raising alarm for {alarm_data}")
        alarm_time = time.time()
        file_name = f'./alarms/Users-{round(alarm_time)}.jpg'
        cv2.imwrite(file_name, face)
        self.files[name] = file_name

        if self.client:
            await self.client.send_file(TELEGRAM_CHANNEL_ID, file_name,
                                        caption=f'Person {name} detected at {time.ctime()}, info: {alarm_data}')

        self.alarm_time[name] = alarm_time

    async def raise_alarm(self, alarm_data, face):
        await self.alarm_queue.put((alarm_data, face))

    def stop(self):
        self.alarm_queue.put((None, None))
        self.alarm_thread.join()

    async def add_detection(self, da: DetectionAggregate, min_frames=5, last_seen_minutes=15):
        detections = da.detections

        for name in detections:
            if detections[name]['count'] < min_frames:
                continue

            if name not in self.agg.detections:
                await self.raise_alarm({'name': name, **detections[name], }, da.faces[name])
                continue

            name_status = self.agg.detections[name]
            if time.time() > name_status['last_frame_time_epoch'] + last_seen_minutes * 60:
                await self.raise_alarm({'name': name, **detections[name], }, da.faces[name])

        self.agg = DetectionAggregate.combine(self.agg, da)

    async def __reply_who(self, _):

        for name, detection in self.agg.detections.items():
            file_name = self.files.get(name)
            message = f'{name}\n' \
                      f'seen first at {time.ctime(detection["first_frame_time_epoch"])}\n' \
                      f'last at {time.ctime(detection["last_frame_time_epoch"])}\n' \
                      f'emotions: {detection["emotions"]}\n'

            if file_name is not None:
                await self.client.send_file(TELEGRAM_CHANNEL_ID, file_name, caption=message)
            else:
                await self.client.send_message(TELEGRAM_CHANNEL_ID, message=message)

    async def __reset_detections(self, event):
        self.agg = DetectionAggregate()
        await event.reply('done')
