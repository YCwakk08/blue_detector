import logging
import queue
import threading
import time
from typing import Union

import cv2


class VideoReader:
    def __init__(self, source: Union[int, str], fps_limit: int) -> None:
        self._source = source
        self._interval = 1.0 / max(fps_limit, 1)
        self._logger = logging.getLogger(__name__)

    def run(self, frame_queue: queue.Queue, stop_event: threading.Event) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            self._logger.error("无法打开视频源: %s", self._source)
            stop_event.set()
            return

        self._logger.info("视频源已打开: %s", self._source)
        last_tick = 0.0

        while not stop_event.is_set():
            now = time.monotonic()
            if now - last_tick < self._interval:
                time.sleep(0.001)
                continue

            ret, frame = cap.read()
            if not ret:
                if isinstance(self._source, str):
                    # 视频文件结束，循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self._logger.warning("读取帧失败，等待重试...")
                    time.sleep(1.0)
                    continue

            last_tick = now
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # 检测线程繁忙时丢帧

        cap.release()
        self._logger.info("视频读取线程已停止")
