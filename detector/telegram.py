import os

# These example values won't work. You must get your own api_id and
# api_hash from https://my.telegram.org, under API Development.
TELEGRAM_API_ID = os.getenv("TELEGRAM_APP_ID")  # 123456
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")  # '8a09818d578d4950bd26a475f765b44e'
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID"))  # -1001485389244

