import queue
import signal
import sys
import threading
import time
from pathlib import Path

import yaml

from api import DetectionAPI
from detector import YOLODetector
from logger import get_logger, setup_logging
from storage import DetectionStorage
from video_reader import VideoReader


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    log_cfg = config.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("file"))
    logger = get_logger("main")
    logger.info("启动 YOLO 视频检测系统")

    # ── 初始化模块 ──────────────────────────────────────────────────────────
    storage = DetectionStorage(
        output_file=config["storage"]["output_file"],
        max_memory_records=config["storage"]["max_memory_records"],
    )
    detector = YOLODetector(
        model_path=config["model"]["path"],
        confidence=config["model"]["confidence"],
        device=config["model"]["device"],
    )
    reader = VideoReader(
        source=config["video"]["source"],
        fps_limit=config["video"]["fps_limit"],
    )

    # ── 共享状态 ────────────────────────────────────────────────────────────
    frame_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    frame_counter = [0]  # 用列表使闭包可以修改

    def get_status() -> dict:
        return {
            "running": not stop_event.is_set(),
            "processed_frames": frame_counter[0],
            "total_records": storage.total,
        }

    # ── 启动 API 服务 ───────────────────────────────────────────────────────
    api = DetectionAPI(
        storage=storage,
        host=config["api"]["host"],
        port=config["api"]["port"],
        get_status=get_status,
    )
    api.start()

    # ── 启动工作线程 ────────────────────────────────────────────────────────
    detect_thread = threading.Thread(
        target=detector.run,
        args=(frame_queue, storage, stop_event, frame_counter),
        name="detector",
        daemon=True,
    )
    reader_thread = threading.Thread(
        target=reader.run,
        args=(frame_queue, stop_event),
        name="reader",
        daemon=True,
    )
    detect_thread.start()
    reader_thread.start()

    # ── 优雅停止 ────────────────────────────────────────────────────────────
    def shutdown(sig, frame):
        logger.info("收到停止信号，正在退出...")
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while not stop_event.is_set():
            time.sleep(2)
            logger.debug(
                "已处理帧: %d  总记录: %d",
                frame_counter[0],
                storage.total,
            )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        detect_thread.join(timeout=5)
        reader_thread.join(timeout=5)
        logger.info("系统已停止，共处理 %d 帧", frame_counter[0])


if __name__ == "__main__":
    main()
