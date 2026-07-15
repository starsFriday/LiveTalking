"""Streaming audio format adapters used by the MiniCPM-o bridge."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np
from av import AudioFrame
from av.audio.resampler import AudioResampler


class FloatRingBuffer:
    """A small allocation-friendly FIFO for mono float32 PCM."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._chunks.clear()
        self._size = 0

    def write(self, samples: np.ndarray) -> None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size:
            self._chunks.append(data.copy())
            self._size += data.size

    def read(self, count: int) -> np.ndarray:
        if count < 0 or count > self._size:
            raise ValueError(f"cannot read {count} samples from {self._size}")
        output = np.empty(count, dtype=np.float32)
        offset = 0
        while offset < count:
            chunk = self._chunks[0]
            take = min(count - offset, chunk.size)
            output[offset : offset + take] = chunk[:take]
            offset += take
            self._size -= take
            if take == chunk.size:
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[take:]
        return output


class StreamingAudioBridge:
    """Convert MiniCPM 24 kHz PCM into LiveTalking 16 kHz/20 ms frames."""

    INPUT_RATE = 24000
    OUTPUT_RATE = 16000
    FRAME_SAMPLES = 320

    def __init__(
        self,
        frame_callback: Callable[[np.ndarray, dict], None],
        max_buffer_ms: int = 2000,
    ) -> None:
        self._frame_callback = frame_callback
        self._resampler = AudioResampler(format="flt", layout="mono", rate=self.OUTPUT_RATE)
        self._buffer = FloatRingBuffer()
        self._max_samples = max(self.FRAME_SAMPLES, self.OUTPUT_RATE * max_buffer_ms // 1000)
        self._response_id: Optional[str] = None
        self._first_frame = False

    @property
    def buffered_ms(self) -> float:
        return self._buffer.size * 1000.0 / self.OUTPUT_RATE

    def reset(self) -> None:
        self._buffer.clear()
        self._resampler = AudioResampler(format="flt", layout="mono", rate=self.OUTPUT_RATE)
        self._response_id = None
        self._first_frame = False

    def push(self, samples_24k: np.ndarray, response_id: Optional[str] = None) -> int:
        samples = np.asarray(samples_24k, dtype=np.float32).reshape(-1)
        if not samples.size:
            return 0
        if response_id != self._response_id:
            self._response_id = response_id
            self._first_frame = True

        frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = self.INPUT_RATE
        for converted in self._resampler.resample(frame):
            self._buffer.write(converted.to_ndarray().reshape(-1))

        if self._buffer.size > self._max_samples:
            raise BufferError(
                f"MiniCPM audio buffer exceeded {self._max_samples * 1000 // self.OUTPUT_RATE} ms"
            )
        return self._emit_complete_frames()

    def finish_response(self) -> int:
        """Flush a short tail and mark the last emitted frame as an end event."""
        for converted in self._resampler.resample(None):
            self._buffer.write(converted.to_ndarray().reshape(-1))
        emitted = self._emit_complete_frames()
        if self._buffer.size:
            tail = self._buffer.read(self._buffer.size)
            padded = np.zeros(self.FRAME_SAMPLES, dtype=np.float32)
            padded[: tail.size] = tail
            self._frame_callback(padded, {"status": "end", "response_id": self._response_id})
            emitted += 1
        elif self._response_id is not None:
            # Keep the end marker on the audio timeline even when resampling
            # happened to produce an exact multiple of 20 ms frames.
            self._frame_callback(
                np.zeros(self.FRAME_SAMPLES, dtype=np.float32),
                {"status": "end", "response_id": self._response_id},
            )
            emitted += 1
        self._response_id = None
        self._first_frame = False
        self._resampler = AudioResampler(format="flt", layout="mono", rate=self.OUTPUT_RATE)
        return emitted

    def _emit_complete_frames(self) -> int:
        emitted = 0
        while self._buffer.size >= self.FRAME_SAMPLES:
            pcm = np.clip(self._buffer.read(self.FRAME_SAMPLES), -1.0, 1.0)
            metadata = {"response_id": self._response_id}
            if self._first_frame:
                metadata["status"] = "start"
                self._first_frame = False
            self._frame_callback(pcm, metadata)
            emitted += 1
        return emitted
