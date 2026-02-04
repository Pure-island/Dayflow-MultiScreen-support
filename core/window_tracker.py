"""
Dayflow Windows - 窗口信息追踪模块
使用 Windows API 获取当前活动窗口的进程名和标题
"""

import logging
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional, Any, cast

logger = logging.getLogger(__name__)

# 尝试导入 Windows 相关库
try:
    import win32gui
    import win32process
    import win32api
    import psutil

    WINDOWS_API_AVAILABLE = True
except ImportError:
    WINDOWS_API_AVAILABLE = False
    win32gui = None
    win32process = None
    win32api = None
    psutil = None
    logger.warning("win32gui/psutil 未安装，窗口追踪功能不可用")


@dataclass
class WindowInfo:
    """窗口信息"""

    app_name: str  # 进程名，如 "chrome.exe"
    window_title: str  # 窗口标题
    process_id: int  # 进程 ID
    window_rect: Optional[tuple] = None  # (left, top, right, bottom)
    monitor_id: Optional[int] = None  # 所在显示器索引

    def get_clean_app_name(self) -> str:
        """获取干净的应用名称（去掉 .exe 后缀）"""
        name = self.app_name
        if name.lower().endswith(".exe"):
            name = name[:-4]
        return name


class WindowTracker:
    """
    窗口追踪器
    获取当前活动窗口的信息
    """

    def __init__(self):
        self._available = WINDOWS_API_AVAILABLE
        if self._available:
            self._try_enable_dpi_awareness()
        # 应用名称映射表，用于标准化常见应用名称
        self._app_name_map = {
            "code": "Visual Studio Code",
            "cursor": "Cursor",
            "chrome": "Google Chrome",
            "msedge": "Microsoft Edge",
            "firefox": "Firefox",
            "wechat": "微信",
            "weixin": "微信",
            "qq": "QQ",
            "dingtalk": "钉钉",
            "feishu": "飞书",
            "lark": "飞书",
            "slack": "Slack",
            "discord": "Discord",
            "telegram": "Telegram",
            "notion": "Notion",
            "obsidian": "Obsidian",
            "typora": "Typora",
            "word": "Microsoft Word",
            "winword": "Microsoft Word",
            "excel": "Microsoft Excel",
            "powerpnt": "Microsoft PowerPoint",
            "outlook": "Microsoft Outlook",
            "notepad": "记事本",
            "notepad++": "Notepad++",
            "sublime_text": "Sublime Text",
            "idea64": "IntelliJ IDEA",
            "pycharm64": "PyCharm",
            "webstorm64": "WebStorm",
            "datagrip64": "DataGrip",
            "explorer": "文件资源管理器",
            "windowsterminal": "Windows Terminal",
            "cmd": "命令提示符",
            "powershell": "PowerShell",
            "spotify": "Spotify",
            "cloudmusic": "网易云音乐",
            "qqmusic": "QQ音乐",
            "potplayer": "PotPlayer",
            "potplayer64": "PotPlayer",
            "vlc": "VLC",
            "steam": "Steam",
            "epicgameslauncher": "Epic Games",
        }

    @property
    def is_available(self) -> bool:
        """检查窗口追踪功能是否可用"""
        return self._available

    def get_active_window(self) -> Optional[WindowInfo]:
        """
        获取当前活动窗口的信息

        Returns:
            WindowInfo 或 None（如果获取失败）
        """
        if (
            not self._available
            or win32gui is None
            or win32process is None
            or psutil is None
        ):
            return None

        win32gui_mod = cast(Any, win32gui)
        win32process_mod = cast(Any, win32process)
        psutil_mod = cast(Any, psutil)

        try:
            # 获取前台窗口句柄
            hwnd = win32gui_mod.GetForegroundWindow()
            if not hwnd:
                return None

            # 获取窗口标题
            window_title = win32gui_mod.GetWindowText(hwnd)

            # 获取窗口矩形
            try:
                window_rect = win32gui_mod.GetWindowRect(hwnd)
            except Exception:
                window_rect = None

            # 获取窗口所在显示器索引
            monitor_id = self._get_monitor_id(hwnd) if window_rect else None

            # 获取进程 ID
            _, pid = win32process_mod.GetWindowThreadProcessId(hwnd)

            # 获取进程名
            try:
                process = psutil_mod.Process(pid)
                app_name = process.name()
            except (psutil_mod.NoSuchProcess, psutil_mod.AccessDenied):
                app_name = "Unknown"

            return WindowInfo(
                app_name=app_name,
                window_title=window_title,
                process_id=pid,
                window_rect=window_rect,
                monitor_id=monitor_id,
            )

        except Exception as e:
            logger.debug(f"获取窗口信息失败: {e}")
            return None

    def _get_monitor_id(self, hwnd) -> Optional[int]:
        """获取窗口所在显示器索引"""
        if win32api is None:
            return None
        win32api_mod = cast(Any, win32api)
        try:
            monitors = win32api_mod.EnumDisplayMonitors()
            target = win32api_mod.MonitorFromWindow(hwnd)
            for idx, (hmon, _hdc, _rect) in enumerate(monitors):
                if hmon == target:
                    return idx
        except Exception:
            return None
        return None

    def _try_enable_dpi_awareness(self):
        """尽量启用 DPI 感知，减少坐标偏差"""
        try:
            import ctypes

            try:
                shcore = ctypes.windll.shcore
                shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    def get_idle_seconds(self) -> float:
        """获取系统空闲秒数（无键鼠输入）"""
        if not self._available:
            return 0.0

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        try:
            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)) == 0:
                return 0.0
            tick = ctypes.windll.kernel32.GetTickCount64()
            idle_ms = tick - lii.dwTime
            return max(idle_ms / 1000.0, 0.0)
        except Exception:
            return 0.0

    def get_friendly_app_name(self, window_info: Optional[WindowInfo]) -> str:
        """
        获取友好的应用名称

        Args:
            window_info: 窗口信息

        Returns:
            友好的应用名称
        """
        if not window_info:
            return "Unknown"

        # 获取干净的进程名（去掉 .exe）
        clean_name = window_info.get_clean_app_name().lower()

        # 查找映射表
        if clean_name in self._app_name_map:
            return self._app_name_map[clean_name]

        # 返回原始名称（首字母大写）
        return window_info.get_clean_app_name().title()


# 全局单例
_tracker: Optional[WindowTracker] = None


def get_tracker() -> WindowTracker:
    """获取窗口追踪器单例"""
    global _tracker
    if _tracker is None:
        _tracker = WindowTracker()
    return _tracker


def get_active_window_info() -> Optional[WindowInfo]:
    """便捷函数：获取当前活动窗口信息"""
    return get_tracker().get_active_window()
