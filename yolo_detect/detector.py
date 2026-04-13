import logging
import queue
import threading
from datetime import datetime
from typing import Any, Dict, List

from storage import DetectionStorage


class YOLODetector:
    def __init__(self, model_path: str, confidence: float, device: str) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._device = device
        self._model = None
        self._logger = logging.getLogger(__name__)

    def _load_model(self) -> None:
        from ultralytics import YOLO  # 延迟导入，避免启动时阻塞
        self._logger.info("加载模型: %s", self._model_path)
        self._model = YOLO(self._model_path)
        self._logger.info("模型加载完成")

    def _infer(self, frame) -> List[Dict[str, Any]]:
        results = self._model(
            frame, conf=self._confidence, device=self._device, verbose=False
        )
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "label": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
                })
        return detections

    def run(
        self,
        frame_queue: queue.Queue,
        storage: DetectionStorage,
        stop_event: threading.Event,
        frame_counter: List[int],
    ) -> None:
        self._load_model()
        self._logger.info("检测线程启动")
        frame_id = 0

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                detections = self._infer(frame)
                storage.save({
                    "frame_id": frame_id,
                    "timestamp": datetime.now().isoformat(),
                    "detections": detections,
                })
                frame_id += 1
                frame_counter[0] = frame_id

                if detections:
                    labels = [d["label"] for d in detections]
                    self._logger.debug("帧 %d: 检测到 %s", frame_id, labels)
            except Exception as e:
                self._logger.error("检测出错: %s", e)

        self._logger.info("检测线程已停止")
