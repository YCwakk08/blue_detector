import logging
import threading
from typing import Callable, Dict, Any

from flask import Flask, jsonify, request

from storage import DetectionStorage


class DetectionAPI:
    def __init__(
        self,
        storage: DetectionStorage,
        host: str,
        port: int,
        get_status: Callable[[], Dict[str, Any]],
    ) -> None:
        self._storage = storage
        self._host = host
        self._port = port
        self._get_status = get_status
        self._logger = logging.getLogger(__name__)
        self._app = Flask(__name__)
        self._register_routes()

    def _register_routes(self) -> None:
        storage = self._storage
        get_status = self._get_status

        @self._app.get("/api/detections")
        def list_detections():
            limit = min(int(request.args.get("limit", 50)), 200)
            offset = max(int(request.args.get("offset", 0)), 0)
            label = request.args.get("label")
            records = storage.get_records(limit=limit, offset=offset, label=label)
            return jsonify({"total": storage.total, "records": records})

        @self._app.get("/api/detections/latest")
        def latest_detection():
            record = storage.get_latest()
            if record is None:
                return jsonify({"error": "暂无记录"}), 404
            return jsonify(record)

        @self._app.get("/api/status")
        def status():
            return jsonify(get_status())

    def start(self) -> threading.Thread:
        t = threading.Thread(
            target=lambda: self._app.run(
                host=self._host,
                port=self._port,
                use_reloader=False,
                threaded=True,
            ),
            name="api",
            daemon=True,
        )
        t.start()
        self._logger.info("API 服务已启动: http://%s:%d", self._host, self._port)
        return t
