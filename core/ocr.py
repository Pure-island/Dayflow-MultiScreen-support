"""
Dayflow Windows - OCR 工具
使用 rapidocr-onnxruntime 进行文本识别
"""

import logging
from typing import List

import cv2

import config

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

logger = logging.getLogger(__name__)


class OcrEngine:
    """OCR 引擎封装"""

    def __init__(self):
        self._engine = None
        if RapidOCR:
            providers = self._select_providers()
            try:
                self._engine = RapidOCR(providers=providers)
                logger.info(f"OCR providers: {providers}")
            except TypeError:
                self._engine = RapidOCR()
                logger.info("OCR providers fallback to default")

    @property
    def available(self) -> bool:
        return self._engine is not None

    def _select_providers(self) -> List[str]:
        device = (config.OCR_DEVICE or "cpu").lower()
        if device != "gpu":
            logger.info("OCR device: cpu")
            return ["CPUExecutionProvider"]

        preferred = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        if ort and hasattr(ort, "get_available_providers"):
            available = ort.get_available_providers()
            logger.info(f"OCR device: gpu, available providers: {available}")
            return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

        logger.info("OCR device: gpu, provider query unavailable")
        return preferred

    def ocr_frames(self, frames: List):
        if not self._engine:
            logger.warning("OCR 引擎不可用")
            return ["" for _ in frames]

        results = []
        for frame in frames:
            try:
                processed = self._preprocess(frame)
                ocr_result, _ = self._engine(processed)
                text = self._join_text(ocr_result)
            except Exception as e:
                logger.warning(f"OCR 识别失败: {e}")
                text = ""
            results.append(text)
        return results

    def _preprocess(self, frame):
        height, width = frame.shape[:2]
        target_width = 1280
        scale = min(target_width / width, 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def _join_text(self, ocr_result):
        if not ocr_result:
            return ""
        lines = []
        for item in ocr_result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = item[1]
                if text:
                    lines.append(str(text))
        return "\n".join(lines)


_engine = None


def get_ocr_engine() -> OcrEngine:
    global _engine
    if _engine is None:
        _engine = OcrEngine()
    return _engine
