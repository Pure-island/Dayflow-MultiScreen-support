"""
Dayflow Windows - 屏幕录制模块
使用 dxcam 实现低功耗 1FPS 录制
"""

import time
import logging
import threading
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Dict, Tuple, Any, cast

import dxcam
import numpy as np
import cv2

win32api = None

try:
    import win32api

    WINDOWS_API_AVAILABLE = True
except ImportError:
    WINDOWS_API_AVAILABLE = False

import config
from core.types import VideoChunk, ChunkStatus
from core.window_tracker import get_tracker, WindowInfo

logger = logging.getLogger(__name__)


class ScreenRecorder:
    """
    屏幕录制器
    - 1 FPS 低功耗录制
    - 每 60 秒自动切片
    - H.264 编码，低码率
    """

    def __init__(
        self,
        fps: Optional[int] = None,
        chunk_duration: Optional[int] = None,
        output_dir: Optional[Path] = None,
        on_chunk_saved: Optional[Callable[[VideoChunk], None]] = None,
    ):
        self.fps = fps or config.RECORD_FPS
        self.chunk_duration = chunk_duration or config.CHUNK_DURATION_SECONDS
        self.output_dir = output_dir or config.CHUNKS_DIR
        self.on_chunk_saved = on_chunk_saved

        # 状态
        self._recording = False
        self._paused = False
        self._camera: Optional[dxcam.DXCamera] = None
        self._cameras: List[dxcam.DXCamera] = []
        self._record_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 当前切片信息
        self._current_writer: Optional[cv2.VideoWriter] = None
        self._current_chunk_path: Optional[Path] = None
        self._current_chunk_start: Optional[datetime] = None
        self._frame_count = 0

        # 窗口追踪
        self._window_tracker = get_tracker()
        self._current_window_records: List[Dict] = []  # 当前切片的窗口记录

        # 多屏布局
        self._monitor_layout: Optional[List[Dict]] = None
        self._canvas_width: Optional[int] = None
        self._canvas_height: Optional[int] = None
        self._canvas_scale: float = 1.0
        self._last_frame_sizes: Dict[int, Tuple[int, int]] = {}
        self._last_frames: Dict[int, np.ndarray] = {}
        self._idle_paused = False

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self):
        """开始录制"""
        if self._recording:
            logger.warning("录制已在进行中")
            return

        logger.info("开始屏幕录制...")

        # 初始化 dxcam
        try:
            output_indices = self._get_output_indices()
            cameras = []
            for idx in output_indices:
                try:
                    cameras.append(dxcam.create(output_idx=idx, output_color="BGR"))
                except Exception as e:
                    logger.warning(f"初始化屏幕 {idx} 失败: {e}")
            if not cameras and output_indices != [0]:
                try:
                    cameras.append(dxcam.create(output_idx=0, output_color="BGR"))
                    logger.info("已回退到屏幕 0 进行录制")
                except Exception as e:
                    logger.warning(f"初始化屏幕 0 失败: {e}")
            self._cameras = cameras
            self._camera = self._cameras[0] if self._cameras else None
        except Exception as e:
            logger.error(f"初始化 dxcam 失败: {e}")
            raise

        if not self._cameras:
            raise RuntimeError("未能初始化任何屏幕录制设备")

        self._recording = True
        self._paused = False
        self._stop_event.clear()

        # 启动录制线程
        self._record_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._record_thread.start()

        logger.info(f"录制已启动 - FPS: {self.fps}, 切片时长: {self.chunk_duration}秒")

    def stop(self):
        """停止录制"""
        if not self._recording:
            return

        logger.info("停止屏幕录制...")

        self._stop_event.set()
        self._recording = False

        # 等待录制线程结束（缩短超时时间）
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=2)
            if self._record_thread.is_alive():
                logger.warning("录制线程未能在超时内停止")

        # 保存当前切片
        try:
            self._finalize_current_chunk()
        except Exception as e:
            logger.error(f"保存切片时出错: {e}")

        # 释放 dxcam
        try:
            for cam in self._cameras:
                try:
                    del cam
                except Exception:
                    pass
            self._cameras = []
            self._camera = None
        except Exception as e:
            logger.error(f"释放相机时出错: {e}")

        logger.info("录制已停止")

    def pause(self):
        """暂停录制"""
        if self._recording and not self._paused:
            self._paused = True
            logger.info("录制已暂停")

    def resume(self):
        """恢复录制"""
        if self._recording and self._paused:
            self._paused = False
            logger.info("录制已恢复")

    def _recording_loop(self):
        """录制主循环"""
        frame_interval = 1.0 / self.fps
        last_frame_time = 0

        while not self._stop_event.is_set():
            current_time = time.time()

            idle_seconds = self._window_tracker.get_idle_seconds()
            if idle_seconds >= config.IDLE_THRESHOLD_SECONDS:
                if not self._idle_paused:
                    logger.info(f"检测到空闲 {idle_seconds:.0f}s，暂停录制")
                    if self._current_writer and self._frame_count > 0:
                        self._finalize_current_chunk()
                    self._idle_paused = True
                time.sleep(0.5)
                continue

            if self._idle_paused:
                logger.info("检测到活动恢复，继续录制并开始新切片")
                self._idle_paused = False
                self._current_chunk_start = None

            # 控制帧率
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.1)  # 短暂休眠以降低 CPU 占用
                continue

            # 暂停检查
            if self._paused:
                time.sleep(0.5)
                continue

            try:
                # 捕获所有屏幕
                frames = []
                for cam in self._cameras:
                    try:
                        frames.append(cam.grab())
                    except Exception:
                        frames.append(None)

                if not any(frame is not None for frame in frames):
                    time.sleep(0.1)
                    continue

                # 先采集窗口信息（在帧捕获时立即采集，确保时间对齐）
                frame_capture_time = datetime.now()
                window_info = self._window_tracker.get_active_window()

                # 合成大画布
                canvas = self._build_canvas(frames)
                if canvas is None:
                    time.sleep(0.1)
                    continue

                # 检查是否需要创建新切片
                if self._should_create_new_chunk():
                    self._finalize_current_chunk()
                    self._create_new_chunk(canvas.shape)

                # 记录窗口信息（使用帧捕获时的时间计算 elapsed）
                if self._current_chunk_start and window_info:
                    elapsed = (
                        frame_capture_time - self._current_chunk_start
                    ).total_seconds()
                    self._current_window_records.append(
                        {
                            "timestamp": elapsed,
                            "app_name": self._window_tracker.get_friendly_app_name(
                                window_info
                            ),
                            "window_title": window_info.window_title,
                            "process_name": window_info.app_name,
                            "monitor_id": window_info.monitor_id,
                            "window_rect": window_info.window_rect,
                        }
                    )

                # 写入帧
                if self._current_writer:
                    self._current_writer.write(canvas)
                    self._frame_count += 1

                last_frame_time = current_time

            except Exception as e:
                logger.error(f"录制帧错误: {e}")
                time.sleep(1)

    def _should_create_new_chunk(self) -> bool:
        """检查是否需要创建新切片"""
        if self._current_chunk_start is None:
            return True

        elapsed = (datetime.now() - self._current_chunk_start).total_seconds()
        return elapsed >= self.chunk_duration

    def _create_new_chunk(self, frame_shape: tuple):
        """创建新的视频切片"""
        timestamp = datetime.now()
        filename = f"chunk_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        self._current_chunk_path = self.output_dir / filename
        self._current_chunk_start = timestamp
        self._frame_count = 0
        self._current_window_records = []  # 重置窗口记录

        # 创建 VideoWriter
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self._current_writer = cv2.VideoWriter(
            str(self._current_chunk_path), fourcc, self.fps, (width, height)
        )

        logger.debug(f"创建新切片: {filename}")

    def _finalize_current_chunk(self):
        """完成当前切片"""
        if self._current_writer is None:
            return

        self._current_writer.release()

        chunk_path = self._current_chunk_path
        if chunk_path and chunk_path.exists():
            end_time = datetime.now()
            start_time = self._current_chunk_start or end_time
            duration = (end_time - start_time).total_seconds()

            # 保存窗口记录到 JSON 文件
            window_records_path = None
            if self._current_window_records or self._monitor_layout:
                window_records_path = chunk_path.with_suffix(".json")
                try:
                    window_records_data = {
                        "layout": self._monitor_layout,
                        "canvas": {
                            "width": self._canvas_width,
                            "height": self._canvas_height,
                            "scale": self._canvas_scale,
                        },
                        "records": self._current_window_records,
                    }
                    with open(window_records_path, "w", encoding="utf-8") as f:
                        json.dump(window_records_data, f, ensure_ascii=False, indent=2)
                    logger.debug(f"窗口记录已保存: {window_records_path.name}")
                except Exception as e:
                    logger.warning(f"保存窗口记录失败: {e}")
                    window_records_path = None

            # 创建切片对象
            chunk = VideoChunk(
                file_path=str(chunk_path),
                start_time=self._current_chunk_start,
                end_time=end_time,
                duration_seconds=duration,
                status=ChunkStatus.PENDING,
                window_records_path=str(window_records_path)
                if window_records_path
                else None,
            )

            logger.info(
                f"切片已保存: {chunk_path.name} ({duration:.1f}秒, {self._frame_count}帧, {len(self._current_window_records)}条窗口记录)"
            )

            # 回调通知
            if self.on_chunk_saved:
                try:
                    self.on_chunk_saved(chunk)
                except Exception as e:
                    logger.error(f"切片保存回调错误: {e}")

        self._current_writer = None
        self._current_chunk_path = None
        self._current_chunk_start = None
        self._frame_count = 0
        self._current_window_records = []

    def _get_output_indices(self) -> List[int]:
        """获取可用输出索引列表"""
        indices: List[int] = []
        info = None
        try:
            if hasattr(dxcam, "output_info"):
                info = dxcam.output_info()
        except Exception:
            info = None

        if isinstance(info, dict):
            outputs = info.get("outputs") or info.get("Outputs")
            if isinstance(outputs, list):
                indices = list(range(len(outputs)))
        elif isinstance(info, list):
            indices = list(range(len(info)))

        if not indices and WINDOWS_API_AVAILABLE and win32api is not None:
            try:
                win32api_mod = cast(Any, win32api)
                indices = list(range(len(win32api_mod.EnumDisplayMonitors())))
            except Exception:
                indices = []

        if not indices:
            indices = [0]

        max_probe = max(1, int(getattr(config, "MAX_OUTPUT_PROBE", 8)))
        if len(indices) > max_probe:
            logger.warning(
                f"输出索引数量过多({len(indices)}), 已限制为前 {max_probe} 个"
            )
            indices = indices[:max_probe]

        return indices

    def _build_canvas(self, frames: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """合成多屏画布，并按需缩放"""
        valid_frames = [f for f in frames if f is not None]
        if not valid_frames:
            return None

        for idx, frame in enumerate(frames):
            if frame is not None:
                self._last_frame_sizes[idx] = (frame.shape[1], frame.shape[0])

        self._ensure_layout(frames)

        if (
            not self._monitor_layout
            or self._canvas_width is None
            or self._canvas_height is None
        ):
            return None

        canvas = np.zeros((self._canvas_height, self._canvas_width, 3), dtype=np.uint8)

        for idx, layout in enumerate(self._monitor_layout):
            if idx >= len(frames):
                continue
            frame = frames[idx]
            if frame is None:
                frame = self._last_frames.get(idx)
            else:
                self._last_frames[idx] = frame
            if frame is None:
                continue
            target_w = layout["width"]
            target_h = layout["height"]
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(
                    frame, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
            x = layout["x"]
            y = layout["y"]
            canvas[y : y + target_h, x : x + target_w] = frame

        if self._canvas_scale < 1.0:
            scaled_w = int(self._canvas_width * self._canvas_scale)
            scaled_h = int(self._canvas_height * self._canvas_scale)
            if scaled_w > 0 and scaled_h > 0:
                canvas = cv2.resize(
                    canvas, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
                )

        return canvas

    def _ensure_layout(self, frames: List[Optional[np.ndarray]]):
        """确保布局与画布信息已准备"""
        if self._monitor_layout and len(self._monitor_layout) == len(frames):
            if self._layout_matches_frames(frames):
                return

        layout = self._get_monitor_layout()
        if layout and len(layout) == len(frames):
            normalized = self._normalize_layout(layout)
            adjusted = self._apply_layout_with_scale(normalized, frames)
            if adjusted:
                self._monitor_layout = adjusted
            else:
                self._monitor_layout = self._create_fallback_layout(frames)
        else:
            self._monitor_layout = self._create_fallback_layout(frames)

        self._update_canvas_info()

    def _layout_matches_frames(self, frames: List[Optional[np.ndarray]]) -> bool:
        if not self._monitor_layout:
            return False
        for idx, layout in enumerate(self._monitor_layout):
            if idx >= len(frames):
                return False
            frame = frames[idx]
            if frame is None:
                continue
            if frame.shape[1] != layout["width"] or frame.shape[0] != layout["height"]:
                return False
        return True

    def _apply_layout_with_scale(
        self, layout: List[Dict], frames: List[Optional[np.ndarray]]
    ) -> Optional[List[Dict]]:
        """根据实际帧尺寸修正布局，并按需缩放坐标"""
        scales_x = []
        scales_y = []
        for idx, item in enumerate(layout):
            size = self._last_frame_sizes.get(idx)
            if size and item["width"] > 0 and item["height"] > 0:
                scales_x.append(size[0] / item["width"])
                scales_y.append(size[1] / item["height"])

        scale_x = self._stable_scale(scales_x)
        scale_y = self._stable_scale(scales_y)

        if scale_x is None or scale_y is None:
            return None

        updated = []
        for idx, item in enumerate(layout):
            size = self._last_frame_sizes.get(idx)
            width = size[0] if size else item["width"]
            height = size[1] if size else item["height"]
            updated.append(
                {
                    "id": item["id"],
                    "x": int(item["x"] * scale_x),
                    "y": int(item["y"] * scale_y),
                    "width": width,
                    "height": height,
                    "primary": item.get("primary", False),
                }
            )

        return updated

    def _stable_scale(self, scales: List[float]) -> Optional[float]:
        if not scales:
            return 1.0
        avg = sum(scales) / len(scales)
        for value in scales:
            if abs(value - avg) > 0.05:
                return None
        return avg

    def _get_monitor_layout(self) -> Optional[List[Dict]]:
        """获取系统显示器布局"""
        layout = self._get_dxcam_layout()
        if layout:
            return layout

        if not WINDOWS_API_AVAILABLE or win32api is None:
            return None

        try:
            win32api_mod = cast(Any, win32api)
            monitors = win32api_mod.EnumDisplayMonitors()
            layout = []
            for idx, (hmon, _hdc, rect) in enumerate(monitors):
                info = win32api_mod.GetMonitorInfo(cast(Any, hmon))
                mon_rect = info.get("Monitor", rect)
                left, top, right, bottom = mon_rect
                layout.append(
                    {
                        "id": idx,
                        "x": left,
                        "y": top,
                        "width": right - left,
                        "height": bottom - top,
                        "primary": bool(info.get("Flags", 0) == 1),
                    }
                )
            return layout
        except Exception:
            return None

    def _get_dxcam_layout(self) -> Optional[List[Dict]]:
        """从 dxcam 输出信息获取布局"""
        info = None
        try:
            if hasattr(dxcam, "output_info"):
                info = dxcam.output_info()
        except Exception:
            return None

        outputs = None
        if isinstance(info, dict):
            outputs = info.get("outputs") or info.get("Outputs")
        elif isinstance(info, list):
            outputs = info

        if not isinstance(outputs, list):
            return None

        layout = []
        for idx, item in enumerate(outputs):
            if not isinstance(item, dict):
                return None
            left = item.get("left", item.get("x", item.get("Left")))
            top = item.get("top", item.get("y", item.get("Top")))
            width = item.get("width", item.get("Width"))
            height = item.get("height", item.get("Height"))
            if left is None or top is None or width is None or height is None:
                return None
            primary = bool(
                item.get("primary") or item.get("Primary") or item.get("is_primary")
            )
            layout.append(
                {
                    "id": idx,
                    "x": int(left),
                    "y": int(top),
                    "width": int(width),
                    "height": int(height),
                    "primary": primary,
                }
            )

        return layout if layout else None

    def _normalize_layout(self, layout: List[Dict]) -> List[Dict]:
        """归一化布局坐标（左上角为 0,0）"""
        min_x = min(m["x"] for m in layout)
        min_y = min(m["y"] for m in layout)

        normalized = []
        for m in layout:
            normalized.append(
                {
                    "id": m["id"],
                    "x": m["x"] - min_x,
                    "y": m["y"] - min_y,
                    "width": m["width"],
                    "height": m["height"],
                    "primary": m.get("primary", False),
                }
            )
        return normalized

    def _create_fallback_layout(self, frames: List[Optional[np.ndarray]]) -> List[Dict]:
        """无法获取系统布局时的兜底布局（横向拼接）"""
        layout = []
        x_offset = 0
        for idx, frame in enumerate(frames):
            if frame is not None:
                height, width = frame.shape[:2]
                self._last_frame_sizes[idx] = (width, height)
            else:
                size = self._last_frame_sizes.get(idx, (1, 1))
                width, height = size
            layout.append(
                {
                    "id": idx,
                    "x": x_offset,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "primary": idx == 0,
                }
            )
            x_offset += width
        return layout

    def _update_canvas_info(self):
        """更新画布尺寸与缩放信息"""
        if not self._monitor_layout:
            self._canvas_width = None
            self._canvas_height = None
            self._canvas_scale = 1.0
            return

        max_x = max(m["x"] + m["width"] for m in self._monitor_layout)
        max_y = max(m["y"] + m["height"] for m in self._monitor_layout)
        self._canvas_width = int(max_x)
        self._canvas_height = int(max_y)

        if self._canvas_width > 0 and config.MAX_CANVAS_WIDTH > 0:
            self._canvas_scale = min(1.0, config.MAX_CANVAS_WIDTH / self._canvas_width)
        else:
            self._canvas_scale = 1.0


class RecordingManager:
    """
    录制管理器
    整合录制器和数据库存储
    """

    def __init__(self, storage_manager=None):
        from database.storage import StorageManager

        self.storage = storage_manager or StorageManager()
        self.recorder = ScreenRecorder(on_chunk_saved=self._on_chunk_saved)

    def _on_chunk_saved(self, chunk: VideoChunk):
        """切片保存回调 - 写入数据库"""
        try:
            chunk_id = self.storage.save_chunk(chunk)
            logger.info(f"切片已入库: ID={chunk_id}")
        except Exception as e:
            logger.error(f"切片入库失败: {e}")

    def start_recording(self):
        """开始录制"""
        self.recorder.start()

    def stop_recording(self):
        """停止录制"""
        self.recorder.stop()

    def pause_recording(self):
        """暂停录制"""
        self.recorder.pause()

    def resume_recording(self):
        """恢复录制"""
        self.recorder.resume()

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    @property
    def is_paused(self) -> bool:
        return self.recorder.is_paused
