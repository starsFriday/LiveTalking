"""No-op TTS provider used when speech is supplied by MiniCPM-o."""

import time

from registry import register
from tts.base_tts import BaseTTS


@register("tts", "nulltts")
class NullTTS(BaseTTS):
    def process_tts(self, quit_event):
        while not quit_event.is_set():
            time.sleep(0.2)
