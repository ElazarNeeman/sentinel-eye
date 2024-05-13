from telethon import TelegramClient, events, sync

# These example values won't work. You must get your own api_id and
# api_hash from https://my.telegram.org, under API Development.
api_id = 123456
api_hash = '123546789abcdef'

client = TelegramClient('session_name', api_id, api_hash)
client.start()

#client.send_message('+972545664107', 'Hello! Talking to you from Telethon')
# name = "Elazar"
# alarm_data = {}
# file_name = "alarms/Users-1715251227.jpg"
# client.send_file('+972545664107',file_name, caption=f'Person {name} detected at {time.ctime()}, info: {alarm_data}')