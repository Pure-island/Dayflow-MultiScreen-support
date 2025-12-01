"""
Dayflow Windows - 主题管理
IDE 风格的亮色/暗色主题
"""
from dataclasses import dataclass
from typing import Optional
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, QObject


@dataclass
class Theme:
    """主题颜色定义"""
    name: str
    
    # 背景色
    bg_primary: str      # 主背景
    bg_secondary: str    # 次背景（卡片、面板）
    bg_tertiary: str     # 第三背景（输入框）
    bg_hover: str        # 悬停背景
    bg_sidebar: str      # 侧边栏
    
    # 边框
    border: str
    border_hover: str
    
    # 文字
    text_primary: str    # 主文字
    text_secondary: str  # 次文字
    text_muted: str      # 弱化文字
    
    # 强调色
    accent: str          # 主强调色
    accent_hover: str    # 强调色悬停
    accent_light: str    # 浅强调色（背景用）
    
    # 功能色
    success: str
    warning: str
    error: str
    
    # 滚动条
    scrollbar: str
    scrollbar_hover: str
    
    # 卡片阴影
    shadow: str


# 暗色主题 - 科技感深色 (类似 VS Code Dark+)
DARK_THEME = Theme(
    name="dark",
    bg_primary="#0D1117",       # GitHub Dark 风格深黑
    bg_secondary="#161B22",     # 卡片背景
    bg_tertiary="#21262D",      # 输入框背景
    bg_hover="#30363D",         # 悬停背景
    bg_sidebar="#010409",       # 侧边栏更深
    border="#30363D",
    border_hover="#484F58",
    text_primary="#FFFFFF",     # 纯白，更清晰
    text_secondary="#E6EDF3",   # 次要文字也更亮
    text_muted="#9CA3AF",       # 弱化文字稍亮
    accent="#58A6FF",           # GitHub 蓝
    accent_hover="#79C0FF",
    accent_light="#388BFD26",   # 带透明度的蓝
    success="#3FB950",
    warning="#D29922",
    error="#F85149",
    scrollbar="#484F58",
    scrollbar_hover="#6E7681",
    shadow="rgba(0, 0, 0, 0.4)",
)


# 亮色主题 - 纯白质感 (类似 VS Code Light+)
LIGHT_THEME = Theme(
    name="light",
    bg_primary="#FFFFFF",       # 纯白主背景
    bg_secondary="#F6F8FA",     # 浅灰卡片
    bg_tertiary="#E8ECF0",      # 输入框/进度条背景稍深
    bg_hover="#EAEEF2",         # 悬停
    bg_sidebar="#F6F8FA",       # 侧边栏
    border="#D0D7DE",           # 边框稍深
    border_hover="#AFB8C1",
    text_primary="#1B1F23",     # 接近纯黑
    text_secondary="#24292F",   # 次要文字也很深
    text_muted="#57606A",       # 弱化文字更深
    accent="#0969DA",           # GitHub 蓝
    accent_hover="#0550AE",
    accent_light="#DDF4FF",     # 浅蓝背景
    success="#1A7F37",
    warning="#9A6700",
    error="#CF222E",
    scrollbar="#AFB8C1",
    scrollbar_hover="#8C959F",
    shadow="rgba(31, 35, 40, 0.12)",
)


class ThemeManager(QObject):
    """主题管理器"""
    
    theme_changed = Signal(object)  # 传递 Theme 对象
    
    _instance: Optional['ThemeManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._current_theme = DARK_THEME
        self._initialized = True
    
    @property
    def current_theme(self) -> Theme:
        return self._current_theme
    
    @property
    def is_dark(self) -> bool:
        return self._current_theme.name == "dark"
    
    def set_theme(self, theme: Theme):
        """设置主题"""
        self._current_theme = theme
        self._apply_global_theme()
        self.theme_changed.emit(theme)
    
    def toggle_theme(self):
        """切换主题"""
        if self.is_dark:
            self.set_theme(LIGHT_THEME)
        else:
            self.set_theme(DARK_THEME)
    
    def _apply_global_theme(self):
        """应用全局样式"""
        app = QApplication.instance()
        if app:
            app.setStyleSheet(self.get_global_stylesheet())
    
    def get_global_stylesheet(self) -> str:
        """生成全局样式表"""
        t = self._current_theme
        return f"""
            /* ===== 全局基础 ===== */
            QMainWindow {{
                background-color: {t.bg_primary};
            }}
            
            QWidget {{
                color: {t.text_primary};
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
            }}
            
            /* ===== 滚动条 ===== */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            QScrollBar:vertical {{
                width: 8px;
                background: transparent;
            }}
            QScrollBar::handle:vertical {{
                background: {t.scrollbar};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {t.scrollbar_hover};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            
            /* ===== 输入框 ===== */
            QLineEdit {{
                background-color: {t.bg_tertiary};
                border: 1px solid {t.border};
                border-radius: 8px;
                padding: 10px 14px;
                color: {t.text_primary};
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border-color: {t.accent};
            }}
            QLineEdit::placeholder {{
                color: {t.text_muted};
            }}
            
            /* ===== 按钮 ===== */
            QPushButton {{
                background-color: {t.bg_tertiary};
                color: {t.text_primary};
                border: 1px solid {t.border};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {t.bg_hover};
                border-color: {t.border_hover};
            }}
            QPushButton:pressed {{
                background-color: {t.bg_tertiary};
            }}
            QPushButton:disabled {{
                background-color: {t.bg_secondary};
                color: {t.text_muted};
                border-color: {t.border};
            }}
            
            /* ===== 进度条 ===== */
            QProgressBar {{
                background-color: {t.bg_tertiary};
                border: none;
                border-radius: 6px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                border-radius: 6px;
            }}
            
            /* ===== 工具提示 ===== */
            QToolTip {{
                background-color: {t.bg_secondary};
                color: {t.text_primary};
                border: 1px solid {t.border};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            
            /* ===== 消息框 ===== */
            QMessageBox {{
                background-color: {t.bg_primary};
            }}
            QMessageBox QLabel {{
                color: {t.text_primary};
            }}
            QMessageBox QPushButton {{
                min-width: 80px;
                padding: 8px 20px;
            }}
            
            /* ===== 菜单 ===== */
            QMenu {{
                background-color: {t.bg_secondary};
                border: 1px solid {t.border};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
                color: {t.text_primary};
            }}
            QMenu::item:selected {{
                background-color: {t.bg_hover};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {t.border};
                margin: 4px 8px;
            }}
        """


# 全局函数
def get_theme_manager() -> ThemeManager:
    """获取主题管理器实例"""
    return ThemeManager()


def get_theme() -> Theme:
    """获取当前主题"""
    return get_theme_manager().current_theme


def is_dark_theme() -> bool:
    """是否为暗色主题"""
    return get_theme_manager().is_dark
