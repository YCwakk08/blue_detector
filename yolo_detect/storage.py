import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional


class DetectionStorage:
    def __init__(self, output_file: str, max_memory_records: int) -> None:
        self._output_file = Path(output_file)
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        self._records: deque = deque(maxlen=max_memory_records)
        self._total = 0
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def save(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)
            self._total += 1
        try:
            with open(self._output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            self._logger.error("写入记录失败: %s", e)

    def get_records(
        self,
        limit: int = 50,
        offset: int = 0,
        label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._records)

        if label:
            records = [
                r for r in records
                if any(d["label"] == label for d in r.get("detections", []))
            ]

        records.reverse()  # newest first
        return records[offset: offset + limit]

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._records[-1] if self._records else None

    @property
    def total(self) -> int:
        return self._total
