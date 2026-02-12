"""
Dayflow Windows - API 交互层
使用 OpenAI 兼容格式调用心流 API
"""

import asyncio
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime

import httpx
import cv2

import config
from core.types import Observation, ActivityCard, AppSite, Distraction
from core.ocr import get_ocr_engine

logger = logging.getLogger(__name__)

# 系统提示词
TRANSCRIBE_SYSTEM_PROMPT = """你是屏幕活动分析助手。根据 OCR 文本和窗口信息，描述用户的具体行为。

返回 JSON 格式：
{
  "observations": [
    {"start_ts": xxx, "end_ts": xxx, "text": "干了什么"}
  ]
}

规则：
- start_ts/end_ts 是相对秒数
- 必须覆盖完整时间范围 [0, duration]，不得留空洞；无法识别用"未能识别"
- 不得输出超出范围的时间戳
- text 只描述行为（写什么代码、看什么内容、做什么操作），不要写应用名称
- 参考窗口标题理解上下文（如文件名、网页标题、聊天对象）
- 不要猜测或编造未出现的事件；没有证据时写"未能识别"
- 必须返回非空的 JSON 内容
- 如需思考过程，请极简
- 只返回 JSON"""

GENERATE_CARDS_SYSTEM_PROMPT = """你是时间管理助手。根据观察记录生成活动卡片。

JSON 格式：
{
  "cards": [
    {
      "category": "编程",
      "title": "Dayflow 项目开发",
      "summary": "实现用户登录功能，编写单元测试",
      "start_ts": 0,
      "end_ts": 900,
      "app_sites": [{"name": "VS Code", "duration_seconds": 5400}],
      "distractions": [],
      "productivity_score": 85
    }
  ]
}

类别定义：
- 编程：写代码、调试、代码审查
- 工作：文档、邮件、项目管理、设计
- 学习：看教程、读文档、做笔记
- 会议：视频会议、语音通话
- 社交：聊天、社交媒体
- 娱乐：视频、游戏、音乐
- 休息：无明显活动
- 其他：无法归类

productivity_score 评分标准：
- 90-100：高度专注的核心工作（编程、写作、设计）
- 70-89：一般工作（邮件、文档、会议）
- 50-69：低效工作（频繁切换、碎片化任务）
- 30-49：轻度娱乐（浏览、社交）
- 0-29：纯娱乐（游戏、视频）

时间规则：
- start_ts/end_ts 为相对秒数（相对于本批次开始时间）
- 必须覆盖完整时间范围 [0, duration]，不得留空洞；无法识别用"未识别"
- 不得输出超出范围的时间戳
- 必须返回非空的 JSON 内容
 - 如需思考过程，请极简

合并规则：连续相同应用且相似活动 → 合并为一张卡片
拆分规则：同一时段内切换不同类型活动 → 拆分为多张卡片

覆盖规则：必须覆盖完整时间范围。若存在无法识别的时间段，输出对应卡片，title 统一为 "未识别"。

只返回 JSON"""


class DayflowBackendProvider:
    """
    心流 API 交互类 (OpenAI 兼容格式)
    使用 Chat Completions 接口进行视频分析
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = config.LLM_TIMEOUT_SECONDS,
    ):
        self.api_base_url = (api_base_url or config.API_BASE_URL).rstrip("/")
        self.api_key = api_key or config.API_KEY
        self.model = model or config.API_MODEL
        self.timeout = timeout

        self._client: Optional[httpx.AsyncClient] = None

    @property
    def headers(self) -> dict:
        """请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout), headers=self.headers
            )
        return self._client

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _is_siliconflow_api(self) -> bool:
        return "siliconflow.cn" in self.api_base_url.lower()

    def _apply_thinking_mode(self, request_body: Dict) -> None:
        think_value = (config.LLM_THINK or "off").lower()
        if think_value == "off":
            return

        if self._is_siliconflow_api():
            request_body["enable_thinking"] = True
            budget_map = {
                "on": 4096,
                "low": 1024,
                "medium": 4096,
                "high": 8192,
            }
            budget = budget_map.get(think_value)
            if budget is not None:
                request_body["thinking_budget"] = budget
            return

        request_body["think"] = True if think_value == "on" else think_value

    def _extract_frames_from_video(
        self, video_path: str, max_frames: int = 8
    ) -> Tuple[List[Tuple[int, "cv2.Mat"]], int]:
        """
        从视频中提取关键帧

        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数

        Returns:
            frames_with_index: [(frame_index, frame)]
            total_frames: 视频总帧数
        """
        frames_with_index = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return frames_with_index, 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return frames_with_index, 0

        # 均匀采样帧
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frames_with_index.append((idx, frame))

        cap.release()
        return frames_with_index, total_frames

    def _encode_frames_to_base64(self, frames: List) -> List[str]:
        frames_base64 = []
        for frame in frames:
            max_w = 768
            max_h = 432
            height, width = frame.shape[:2]
            scale = min(max_w / width, max_h / height, 1.0)
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            base64_image = base64.b64encode(buffer).decode("utf-8")
            frames_base64.append(base64_image)
        return frames_base64

    async def _chat_completion(
        self, messages: List[dict], temperature: float = 0.3
    ) -> str:
        """
        调用 Chat Completions API

        Args:
            messages: 消息列表
            temperature: 温度参数

        Returns:
            str: 模型返回的内容
        """
        client = await self._get_client()

        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }

        self._apply_thinking_mode(request_body)

        max_retries = max(0, int(getattr(config, "LLM_MAX_RETRIES", 2)))
        retry_delay = float(getattr(config, "LLM_RETRY_DELAY_SECONDS", 1.0))
        if retry_delay < 0:
            retry_delay = 0.0
        for attempt in range(max_retries + 1):
            try:
                response = await client.post(
                    f"{self.api_base_url}/chat/completions", json=request_body
                )
                response.raise_for_status()

                result = response.json()
                if isinstance(result, dict) and result.get("error"):
                    logger.error(f"API 返回错误: {result.get('error')}")
                    raise ValueError("API response contains error")

                choices = result.get("choices") if isinstance(result, dict) else None
                if not choices:
                    logger.error(f"API 响应缺少 choices: {response.text[:500]}")
                    raise ValueError("API response missing choices")

                message = (
                    choices[0].get("message") if isinstance(choices[0], dict) else None
                )
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                    )

                if not content and isinstance(message, dict):
                    reasoning = (
                        message.get("reasoning")
                        or message.get("thinking")
                        or message.get("reasoning_content")
                    )
                    if reasoning:
                        logger.info("API 返回空 content，使用 reasoning 字段")
                        content = reasoning

                if not content:
                    logger.warning(
                        "API 返回空内容: model=%s response=%s",
                        self.model,
                        response.text[:500],
                    )
                    return ""

                return content

            except httpx.RequestError as e:
                request_url = getattr(e.request, "url", "unknown")
                if attempt < max_retries:
                    logger.warning(
                        "API 请求异常重试(%d/%d): %s %r - %s",
                        attempt + 1,
                        max_retries,
                        type(e).__name__,
                        e,
                        request_url,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                logger.error(f"API 请求异常: {type(e).__name__} {e!r} - {request_url}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"API 请求失败: {e.response.status_code} - {e.response.text}"
                )
                raise
            except Exception as e:
                logger.error(f"API 请求异常: {type(e).__name__} {e!r}")
                raise

        raise RuntimeError("API 请求失败，已耗尽重试次数")

    async def transcribe_video(
        self,
        video_path: str,
        duration: float,
        prompt: Optional[str] = None,
        window_records: Optional[List[Dict]] = None,
        layout_info: Optional[Dict] = None,
    ) -> List[Observation]:
        """
        分析视频切片，获取观察记录

        Args:
            video_path: 视频文件路径
            duration: 视频时长（秒）
            prompt: 额外提示词（可选）
            window_records: 窗口记录列表（可选）

        Returns:
            List[Observation]: 观察记录列表
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 提取视频帧
        frames_with_index, total_frames = self._extract_frames_from_video(
            video_path, max_frames=8
        )
        if not frames_with_index:
            logger.warning(f"无法从视频提取帧: {video_path}")
            return []

        # 构建布局与窗口信息文本（包含窗口标题）
        layout_text = ""
        window_text = ""

        if layout_info:
            layout = (
                layout_info.get("layout") if isinstance(layout_info, dict) else None
            )
            canvas = (
                layout_info.get("canvas") if isinstance(layout_info, dict) else None
            )
            if layout:
                layout_text += "\n\n屏幕布局：\n"
                for item in layout:
                    primary_text = "主屏" if item.get("primary") else ""
                    layout_text += (
                        f"- 屏幕{item.get('id')}: {item.get('width')}x{item.get('height')}"
                        f" (x={item.get('x')}, y={item.get('y')}) {primary_text}\n"
                    )
            if canvas and canvas.get("scale"):
                layout_text += (
                    f"\n画布信息: {canvas.get('width')}x{canvas.get('height')}, "
                    f"缩放比例 {canvas.get('scale'):.2f}\n"
                )
        if window_records:
            window_text = "\n\n窗口信息：\n"
            # 按时间段聚合相同的应用
            current_app = None
            current_title = None
            current_start = 0
            current_monitor = None
            for record in window_records:
                app_name = record.get("app_name", "Unknown")
                window_title = record.get("window_title", "")
                monitor_id = record.get("monitor_id")
                if (
                    app_name != current_app
                    or window_title != current_title
                    or monitor_id != current_monitor
                ):
                    if current_app:
                        title_part = f": {current_title}" if current_title else ""
                        monitor_part = (
                            f" (屏幕{current_monitor})"
                            if current_monitor is not None
                            else ""
                        )
                        window_text += (
                            f"- [{current_start:.0f}s - {record['timestamp']:.0f}s] "
                            f"{current_app}{title_part}{monitor_part}\n"
                        )
                    current_app = app_name
                    current_title = window_title
                    current_monitor = monitor_id
                    current_start = record.get("timestamp", 0)
            # 添加最后一个
            if current_app:
                title_part = f": {current_title}" if current_title else ""
                monitor_part = (
                    f" (屏幕{current_monitor})" if current_monitor is not None else ""
                )
                window_text += (
                    f"- [{current_start:.0f}s - {duration:.0f}s] "
                    f"{current_app}{title_part}{monitor_part}\n"
                )

        window_info_text = layout_text + window_text

        analysis_mode = (config.ANALYSIS_MODE or "ocr").lower()

        # 构建消息内容
        content = []
        if analysis_mode == "ocr":
            ocr_engine = get_ocr_engine()
            frames = [frame for _, frame in frames_with_index]
            ocr_texts = ocr_engine.ocr_frames(frames)
            ocr_lines = []
            total_frames_safe = max(total_frames - 1, 1)
            for (frame_idx, _frame), text in zip(frames_with_index, ocr_texts):
                text = text.strip()
                if not text:
                    continue
                timestamp = duration * (frame_idx / total_frames_safe)
                ocr_lines.append(f"[{timestamp:.0f}s] {text}")
            ocr_text = "\n".join(ocr_lines)
            logger.info(
                "OCR 提取完成: frames=%d chars=%d",
                len(frames_with_index),
                len(ocr_text),
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"以下是一段 {duration:.0f} 秒屏幕录制的 OCR 文本与窗口信息，请分析用户的活动。"
                        f"时间范围为 [0, {duration:.0f}] 秒，必须完整覆盖。"
                        f"\n\nOCR 文本：\n{ocr_text}\n{window_info_text}{prompt or ''}"
                    ),
                }
            )
        else:
            frames = [frame for _, frame in frames_with_index]
            frames_base64 = self._encode_frames_to_base64(frames)
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"以下是一段 {duration:.0f} 秒屏幕录制的 {len(frames_base64)} 个关键帧，请分析用户的活动。"
                        f"时间范围为 [0, {duration:.0f}] 秒，必须完整覆盖。"
                        f"{window_info_text}{prompt or ''}"
                    ),
                }
            )

            for frame_base64 in frames_base64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low",
                        },
                    }
                )

        max_parse_retries = max(0, int(getattr(config, "LLM_PARSE_RETRIES", 2)))
        last_reason = ""
        for attempt in range(max_parse_retries + 1):
            try:
                request_content = content
                if attempt > 0:
                    reason_text = f" reason={last_reason}" if last_reason else ""
                    retry_tag = (
                        f"重试标记: {attempt}/{max_parse_retries}"
                        f" {int(time.time() * 1000)}{reason_text}"
                    )
                    request_content = (
                        [{"type": "text", "text": retry_tag}]
                        + content
                        + [{"type": "text", "text": retry_tag}]
                    )

                messages = [
                    {"role": "system", "content": TRANSCRIBE_SYSTEM_PROMPT},
                    {"role": "user", "content": request_content},
                ]

                response_text = await self._chat_completion(messages)
                logger.info(
                    "视频分析响应: chars=%d preview=%s",
                    len(response_text),
                    response_text,
                )
                observations, parsed = self._parse_observations_from_text(
                    response_text, duration
                )
                logger.info("视频分析解析: observations=%d", len(observations))

                if (
                    not parsed or len(observations) == 0
                ) and attempt < max_parse_retries:
                    if not parsed:
                        last_reason = "parse_failed"
                    elif len(observations) == 0:
                        last_reason = "empty_result"
                    logger.warning(
                        "视频分析解析失败，正在重试 LLM 请求(%d/%d)",
                        attempt + 1,
                        max_parse_retries,
                    )
                    continue

                # 后处理：用真实窗口信息覆盖 AI 返回的 app_name
                if window_records and observations:
                    observations = self._apply_window_records(
                        observations, window_records, duration
                    )

                return observations
            except Exception as e:
                last_reason = f"error={type(e).__name__}"
                logger.error(f"视频分析失败: {e}")
                return []

        return []

    def _apply_window_records(
        self,
        observations: List[Observation],
        window_records: List[Dict],
        duration: float,
    ) -> List[Observation]:
        """
        用真实窗口记录覆盖 AI 返回的 app_name

        根据时间戳匹配，找到每个 observation 对应时间段内使用最多的应用
        """
        if not window_records:
            return observations

        # 预处理：构建时间段到应用的映射
        # 格式: [(start_ts, end_ts, app_name, window_title), ...]
        time_segments = []
        current_app = None
        current_title = None
        current_start = 0

        for record in window_records:
            app_name = record.get("app_name", "Unknown")
            window_title = record.get("window_title", "")
            timestamp = record.get("timestamp", 0)

            if app_name != current_app:
                if current_app:
                    time_segments.append(
                        (current_start, timestamp, current_app, current_title)
                    )
                current_app = app_name
                current_title = window_title
                current_start = timestamp

        # 添加最后一个时间段
        if current_app:
            time_segments.append((current_start, duration, current_app, current_title))

        # 为每个 observation 找到对应的应用
        for obs in observations:
            obs_start = obs.start_ts
            obs_end = obs.end_ts

            # 统计这个时间段内各应用的占用时长
            app_durations: Dict[str, float] = {}
            app_titles: Dict[str, str] = {}

            for seg_start, seg_end, app_name, window_title in time_segments:
                # 计算重叠时间
                overlap_start = max(obs_start, seg_start)
                overlap_end = min(obs_end, seg_end)

                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    app_durations[app_name] = (
                        app_durations.get(app_name, 0) + overlap_duration
                    )
                    if app_name not in app_titles:
                        app_titles[app_name] = window_title

            # 找到占用时间最长的应用
            if app_durations:
                main_app = max(app_durations.items(), key=lambda item: item[1])[0]
                obs.app_name = main_app
                obs.window_title = app_titles.get(main_app, obs.window_title)
                logger.debug(
                    f"后处理: [{obs_start:.0f}s-{obs_end:.0f}s] app_name -> {main_app}"
                )

        return observations

    async def generate_activity_cards(
        self,
        observations: List[Observation],
        context_cards: Optional[List[ActivityCard]] = None,
        start_time: Optional[datetime] = None,
        prompt: Optional[str] = None,
    ) -> List[ActivityCard]:
        """
        根据观察记录生成时间轴卡片

        Args:
            observations: 观察记录列表
            context_cards: 前序卡片（用于上下文）
            start_time: 开始时间
            prompt: 额外提示词（可选）

        Returns:
            List[ActivityCard]: 活动卡片列表
        """
        if not observations:
            return []

        # 构建观察记录文本
        obs_text = "观察记录：\n"
        for obs in observations:
            obs_text += f"- [{obs.start_ts:.0f}s - {obs.end_ts:.0f}s] {obs.text}"
            if obs.app_name:
                obs_text += f" (应用: {obs.app_name})"
            obs_text += "\n"

        # 添加时间上下文
        if start_time:
            obs_text += f"\n录制开始时间: {start_time.isoformat()}"

        obs_text += "\n必须覆盖完整时间范围 [0, duration]，不得留空洞。"

        if context_cards:
            obs_text += "\n\n前序活动卡片：\n"
            for card in context_cards:
                obs_text += f"- [{card.start_time}] {card.category}: {card.title}"
                if card.summary:
                    obs_text += f" | {card.summary}"
                obs_text += "\n"

        if prompt:
            obs_text += f"\n{prompt}"

        max_parse_retries = max(0, int(getattr(config, "LLM_PARSE_RETRIES", 2)))
        last_reason = ""
        for attempt in range(max_parse_retries + 1):
            try:
                request_text = obs_text
                if attempt > 0:
                    reason_text = f" reason={last_reason}" if last_reason else ""
                    retry_tag = (
                        f"重试标记: {attempt}/{max_parse_retries}"
                        f" {int(time.time() * 1000)}{reason_text}"
                    )
                    request_text = f"{retry_tag}\n{obs_text}\n{retry_tag}"

                messages = [
                    {"role": "system", "content": GENERATE_CARDS_SYSTEM_PROMPT},
                    {"role": "user", "content": request_text},
                ]

                response_text = await self._chat_completion(messages)
                logger.info(
                    "卡片生成响应: chars=%d preview=%s",
                    len(response_text),
                    response_text,
                )
                cards, parsed = self._parse_cards_from_text(response_text, start_time)
                logger.info("卡片生成解析: cards=%d", len(cards))
                if parsed and len(cards) > 0:
                    return cards
                if attempt < max_parse_retries:
                    if not parsed:
                        last_reason = "parse_failed"
                    elif len(cards) == 0:
                        last_reason = "empty_result"
                    logger.warning(
                        "卡片解析失败，正在重试 LLM 请求(%d/%d)",
                        attempt + 1,
                        max_parse_retries,
                    )
                    continue
                return []
            except Exception as e:
                last_reason = f"error={type(e).__name__}"
                logger.error(f"卡片生成失败: {e}")
                return []

        return []

    def _parse_observations_from_text(
        self, text: str, duration: float
    ) -> Tuple[List[Observation], bool]:
        """从文本响应中解析观察记录"""
        observations = []

        try:
            stripped = text.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                data = json.loads(self._sanitize_json_text(stripped))
                if isinstance(data, dict):
                    items = data.get("observations", [])
                    observations.extend(self._build_observations(items, duration))
                elif isinstance(data, list):
                    observations.extend(self._build_observations(data, duration))
            else:
                array_match = re.search(r"\[[\s\S]*\]", text)
                if array_match:
                    data = json.loads(self._sanitize_json_text(array_match.group()))
                    if isinstance(data, list):
                        observations.extend(self._build_observations(data, duration))
                else:
                    json_match = re.search(r"\{[\s\S]*\}", text)
                    if json_match:
                        data = json.loads(self._sanitize_json_text(json_match.group()))
                        if isinstance(data, dict):
                            items = data.get("observations", [])
                            observations.extend(
                                self._build_observations(items, duration)
                            )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}, 原文: {text[:200]}")
            # 如果 JSON 解析失败，创建一个基于整段文本的观察记录
            observations.append(
                Observation(start_ts=0, end_ts=duration, text=text[:500])
            )
            return observations, False
        except Exception as e:
            logger.warning(f"解析观察记录失败: {type(e).__name__} {e!r}")
            return observations, False

        if not observations:
            return observations, False

        return observations, True

    def _build_observations(
        self, items: List[Dict], duration: float
    ) -> List[Observation]:
        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            results.append(
                Observation(
                    start_ts=float(item.get("start_ts", 0)),
                    end_ts=float(item.get("end_ts", duration)),
                    text=item.get("text", ""),
                    app_name=item.get("app_name"),
                    window_title=item.get("window_title"),
                )
            )
        return results

    def _parse_cards_from_text(
        self, text: str, start_time: Optional[datetime]
    ) -> Tuple[List[ActivityCard], bool]:
        """从文本响应中解析活动卡片"""
        cards = []

        try:
            items = self._extract_cards_items(text)
            if not items:
                logger.warning("卡片 JSON 为空: %s", text[:200])

            for item in items:
                # 解析时间
                card_start = None
                card_end = None
                relative_start = None
                relative_end = None

                if item.get("start_ts") is not None:
                    try:
                        relative_start = float(item.get("start_ts") or 0)
                    except Exception:
                        relative_start = None
                if item.get("end_ts") is not None:
                    try:
                        relative_end = float(item.get("end_ts") or 0)
                    except Exception:
                        relative_end = None

                if relative_start is None and item.get("start_time"):
                    try:
                        card_start = datetime.fromisoformat(
                            item["start_time"].replace("Z", "+00:00")
                        )
                    except Exception:
                        card_start = start_time

                if relative_end is None and item.get("end_time"):
                    try:
                        card_end = datetime.fromisoformat(
                            item["end_time"].replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

                # 解析应用列表
                app_sites = []
                for app in item.get("app_sites", []):
                    app_sites.append(
                        AppSite(
                            name=app.get("name", ""),
                            duration_seconds=app.get("duration_seconds", 0),
                        )
                    )

                # 解析分心记录
                distractions = []
                for dist in item.get("distractions", []):
                    distractions.append(
                        Distraction(
                            description=dist.get("description", ""),
                            timestamp=dist.get("timestamp", 0),
                            duration_seconds=dist.get("duration_seconds", 0),
                        )
                    )

                card = ActivityCard(
                    category=item.get("category", "其他"),
                    title=item.get("title", "未命名活动"),
                    summary=item.get("summary", ""),
                    start_time=card_start,
                    end_time=card_end,
                    app_sites=app_sites,
                    distractions=distractions,
                    productivity_score=float(item.get("productivity_score") or 0),
                )
                if relative_start is not None:
                    setattr(card, "_relative_start", relative_start)
                if relative_end is not None:
                    setattr(card, "_relative_end", relative_end)
                cards.append(card)

        except json.JSONDecodeError as e:
            logger.warning(f"卡片 JSON 解析失败: {e}")
            return [], False
        except Exception as e:
            logger.warning(f"解析卡片失败: {type(e).__name__} {e!r}")
            return [], False

        if not cards:
            return cards, False

        return cards, True

    def _extract_cards_items(self, text: str) -> List[Dict]:
        stripped = text.strip()
        candidates = []

        if stripped.startswith("[") or stripped.startswith("{"):
            candidates.append(stripped)

        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            candidates.append(array_match.group())

        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            candidates.append(obj_match.group())

        for raw in candidates:
            try:
                sanitized = self._sanitize_card_math(raw)
                data = json.loads(self._sanitize_json_text(sanitized))
            except Exception:
                data = None

            if data is None:
                trimmed = self._trim_incomplete_json(raw)
                if trimmed:
                    try:
                        trimmed = self._sanitize_card_math(trimmed)
                        data = json.loads(self._sanitize_json_text(trimmed))
                    except Exception:
                        data = None
            if data is None:
                continue
            if isinstance(data, dict):
                items = data.get("cards", [])
                if isinstance(items, list):
                    return items
            if isinstance(data, list):
                return data

        return []

    def _sanitize_card_math(self, text: str) -> str:
        def repl(match: re.Match) -> str:
            left = int(match.group(1))
            right = int(match.group(2))
            return f'"duration_seconds": {left - right}'

        return re.sub(
            r"\"duration_seconds\"\s*:\s*(\d+)\s*-\s*(\d+)",
            repl,
            text,
        )

    def _trim_incomplete_json(self, text: str) -> Optional[str]:
        last_obj = text.rfind("}")
        last_arr = text.rfind("]")
        end = max(last_obj, last_arr)
        if end == -1:
            return None
        return text[: end + 1]

    def _sanitize_json_text(self, text: str) -> str:
        """清理 JSON 中未转义的控制字符"""
        result = []
        in_string = False
        escape = False

        for ch in text:
            if escape:
                result.append(ch)
                escape = False
                continue

            if ch == "\\":
                result.append(ch)
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                result.append(ch)
                continue

            if in_string and ord(ch) < 32:
                if ch == "\n":
                    result.append("\\n")
                elif ch == "\r":
                    result.append("\\r")
                elif ch == "\t":
                    result.append("\\t")
                else:
                    result.append(f"\\u{ord(ch):04x}")
                continue

            result.append(ch)

        return "".join(result)

    async def health_check(self) -> bool:
        """检查 API 连接状态"""
        try:
            messages = [{"role": "user", "content": "hi"}]
            await self._chat_completion(messages)
            return True
        except Exception as e:
            logger.warning(f"API 健康检查失败: {e}")
            return False

    async def test_connection(self) -> tuple[bool, str]:
        """
        测试 API 连接

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        if not self.api_key:
            return False, "API Key 未配置"

        try:
            messages = [{"role": "user", "content": "你好，请回复'测试成功'"}]
            response = await self._chat_completion(messages)
            return True, f"连接成功！模型: {self.model}\n回复: {response[:100]}"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP 错误 {e.response.status_code}: {e.response.text[:200]}"
        except httpx.ConnectError:
            return False, "连接失败：无法连接到服务器"
        except httpx.TimeoutException:
            return False, "连接超时"
        except Exception as e:
            return False, f"错误: {str(e)}"


# 便捷函数：同步调用
def transcribe_video_sync(
    video_path: str, duration: float, **kwargs
) -> List[Observation]:
    """同步版本的视频分析"""
    provider = DayflowBackendProvider(**kwargs)
    try:
        return asyncio.run(provider.transcribe_video(video_path, duration))
    finally:
        asyncio.run(provider.close())


def generate_cards_sync(
    observations: List[Observation],
    context_cards: Optional[List[ActivityCard]] = None,
    **kwargs,
) -> List[ActivityCard]:
    """同步版本的卡片生成"""
    provider = DayflowBackendProvider(**kwargs)
    try:
        return asyncio.run(
            provider.generate_activity_cards(observations, context_cards)
        )
    finally:
        asyncio.run(provider.close())
