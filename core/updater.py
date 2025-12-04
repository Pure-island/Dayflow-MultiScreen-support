"""
Dayflow - 自动更新模块
支持版本检查、多源下载、后台更新
"""
import os
import sys
import json
import logging
import threading
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

import httpx

import config

logger = logging.getLogger(__name__)

# 下载源列表（按优先级排序）
DOWNLOAD_MIRRORS = [
    # GitHub 原始地址
    "https://github.com/{repo}/releases/download/{tag}/{filename}",
    # 镜像加速源
    "https://mirror.ghproxy.com/https://github.com/{repo}/releases/download/{tag}/{filename}",
    "https://ghproxy.net/https://github.com/{repo}/releases/download/{tag}/{filename}",
    "https://gh.ddlc.top/https://github.com/{repo}/releases/download/{tag}/{filename}",
]

# GitHub API
GITHUB_API_URL = "https://api.github.com/repos/{repo}/releases/latest"


@dataclass
class UpdateInfo:
    """更新信息"""
    has_update: bool = False
    current_version: str = ""
    latest_version: str = ""
    download_url: str = ""
    release_notes: str = ""
    file_size: int = 0
    filename: str = ""


class UpdateChecker:
    """版本检查器"""
    
    def __init__(self):
        self.repo = config.GITHUB_REPO
        self.current_version = config.VERSION
    
    def check(self) -> UpdateInfo:
        """
        检查是否有新版本
        
        Returns:
            UpdateInfo: 更新信息
        """
        info = UpdateInfo(current_version=self.current_version)
        
        try:
            url = GITHUB_API_URL.format(repo=self.repo)
            
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, follow_redirects=True)
                resp.raise_for_status()
                data = resp.json()
            
            # 解析版本号（去掉 v 前缀）
            latest_tag = data.get('tag_name', '')
            info.latest_version = latest_tag.lstrip('v')
            info.release_notes = data.get('body', '')
            
            # 比较版本
            if self._compare_versions(info.latest_version, self.current_version) > 0:
                info.has_update = True
                
                # 查找下载链接（优先 ZIP，其次 EXE）
                for asset in data.get('assets', []):
                    name_lower = asset['name'].lower()
                    if name_lower.endswith('.zip'):
                        info.download_url = asset['browser_download_url']
                        info.file_size = asset.get('size', 0)
                        info.filename = asset['name']
                        break
                    elif name_lower.endswith('.exe') and not info.filename:
                        info.download_url = asset['browser_download_url']
                        info.file_size = asset.get('size', 0)
                        info.filename = asset['name']
            
            logger.info(f"版本检查完成: 当前 {self.current_version}, 最新 {info.latest_version}")
            return info
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"GitHub API 请求失败: {e.response.status_code}")
            return info
        except Exception as e:
            logger.warning(f"检查更新失败: {e}")
            return info
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        比较版本号
        
        Returns:
            1 if v1 > v2, -1 if v1 < v2, 0 if equal
        """
        try:
            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]
            
            # 补齐长度
            while len(parts1) < len(parts2):
                parts1.append(0)
            while len(parts2) < len(parts1):
                parts2.append(0)
            
            for a, b in zip(parts1, parts2):
                if a > b:
                    return 1
                elif a < b:
                    return -1
            return 0
        except:
            return 0


class UpdateDownloader:
    """更新下载器（支持多源下载）"""
    
    def __init__(
        self,
        update_info: UpdateInfo,
        on_progress: Optional[Callable[[float], None]] = None,
        on_complete: Optional[Callable[[bool, str], None]] = None
    ):
        self.update_info = update_info
        self.on_progress = on_progress
        self.on_complete = on_complete
        
        # 下载目录
        self.pending_dir = config.APP_DATA_DIR / "pending_update"
        self.is_zip = update_info.filename.lower().endswith('.zip') if update_info.filename else False
        
        if self.is_zip:
            self.target_path = self.pending_dir / update_info.filename
        else:
            self.target_path = self.pending_dir / "Dayflow_new.exe"
        
        self._cancelled = False
    
    def start(self):
        """在后台线程开始下载"""
        thread = threading.Thread(target=self._download, daemon=True)
        thread.start()
    
    def cancel(self):
        """取消下载"""
        self._cancelled = True
    
    def _download(self):
        """执行下载（尝试多个源）"""
        try:
            # 创建目录
            self.pending_dir.mkdir(parents=True, exist_ok=True)
            
            # 解析版本 tag
            tag = f"v{self.update_info.latest_version}"
            filename = self.update_info.filename or "Dayflow.exe"
            
            # 尝试多个下载源
            last_error = ""
            for mirror_template in DOWNLOAD_MIRRORS:
                if self._cancelled:
                    return
                
                url = mirror_template.format(
                    repo=config.GITHUB_REPO,
                    tag=tag,
                    filename=filename
                )
                
                logger.info(f"尝试下载源: {url[:70]}...")
                
                try:
                    success = self._download_from_url(url)
                    if success:
                        # 如果是 ZIP，解压
                        if self.is_zip:
                            self._extract_zip()
                        
                        # 下载成功，写入更新信息
                        self._save_update_info()
                        
                        if self.on_complete:
                            self.on_complete(True, "")
                        return
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"下载源失败: {e}")
                    continue
            
            # 所有源都失败
            if self.on_complete:
                self.on_complete(False, f"所有下载源都失败: {last_error}")
                
        except Exception as e:
            logger.error(f"下载过程出错: {e}")
            if self.on_complete:
                self.on_complete(False, str(e))
    
    def _download_from_url(self, url: str) -> bool:
        """从指定 URL 下载"""
        temp_path = self.target_path.with_suffix('.tmp')
        
        try:
            with httpx.stream('GET', url, follow_redirects=True, timeout=300) as response:
                response.raise_for_status()
                
                total = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        if self._cancelled:
                            temp_path.unlink(missing_ok=True)
                            return False
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if self.on_progress and total > 0:
                            self.on_progress(downloaded / total * 100)
            
            # 下载完成，重命名
            if temp_path.exists():
                if self.target_path.exists():
                    self.target_path.unlink()
                temp_path.rename(self.target_path)
                return True
            
            return False
            
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise e
    
    def _extract_zip(self):
        """解压 ZIP 文件"""
        extract_dir = self.pending_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"正在解压: {self.target_path}")
        
        with zipfile.ZipFile(self.target_path, 'r') as zf:
            zf.extractall(extract_dir)
        
        # 查找解压后的 Dayflow.exe
        # 可能在根目录或子目录中
        exe_path = None
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.lower() == 'dayflow.exe':
                    exe_path = Path(root) / f
                    break
            if exe_path:
                break
        
        if exe_path:
            # 移动到 pending_dir/Dayflow_new.exe
            new_exe_path = self.pending_dir / "Dayflow_new.exe"
            shutil.move(str(exe_path), str(new_exe_path))
            
            # 复制同目录下的其他文件（DLL 等依赖）
            exe_dir = exe_path.parent
            for item in exe_dir.iterdir():
                if item.is_file() and item != exe_path:
                    dest = self.pending_dir / item.name
                    shutil.copy2(str(item), str(dest))
                elif item.is_dir():
                    dest_dir = self.pending_dir / item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(str(item), str(dest_dir))
            
            logger.info(f"解压完成，找到 EXE: {new_exe_path}")
        else:
            raise FileNotFoundError("ZIP 中未找到 Dayflow.exe")
        
        # 清理
        self.target_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
    
    def _save_update_info(self):
        """保存更新信息"""
        info = {
            'version': self.update_info.latest_version,
            'ready': True,
            'current_exe_path': sys.executable,
            'release_notes': self.update_info.release_notes
        }
        info_path = self.pending_dir / 'update_info.json'
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        logger.info(f"更新已准备就绪: v{self.update_info.latest_version}")


class UpdateManager:
    """更新管理器 - 统一入口"""
    
    def __init__(self):
        self.checker = UpdateChecker()
        self.downloader: Optional[UpdateDownloader] = None
        self.update_info: Optional[UpdateInfo] = None
        self.pending_dir = config.APP_DATA_DIR / "pending_update"
    
    def check_update(self) -> UpdateInfo:
        """检查更新"""
        self.update_info = self.checker.check()
        return self.update_info
    
    def start_download(
        self,
        on_progress: Optional[Callable[[float], None]] = None,
        on_complete: Optional[Callable[[bool, str], None]] = None
    ):
        """开始下载更新"""
        if not self.update_info or not self.update_info.has_update:
            if on_complete:
                on_complete(False, "没有可用更新")
            return
        
        self.downloader = UpdateDownloader(
            self.update_info,
            on_progress=on_progress,
            on_complete=on_complete
        )
        self.downloader.start()
    
    def cancel_download(self):
        """取消下载"""
        if self.downloader:
            self.downloader.cancel()
    
    def has_pending_update(self) -> bool:
        """检查是否有待安装的更新"""
        info_path = self.pending_dir / 'update_info.json'
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                return info.get('ready', False)
            except:
                pass
        return False
    
    def get_pending_update_info(self) -> Optional[dict]:
        """获取待安装更新的信息"""
        info_path = self.pending_dir / 'update_info.json'
        if info_path.exists():
            try:
                return json.loads(info_path.read_text())
            except:
                pass
        return None
    
    def apply_update(self) -> bool:
        """
        应用更新（启动 updater 并退出主程序）
        
        Returns:
            bool: 是否成功启动更新程序
        """
        import subprocess
        
        # 检查新版本文件
        new_exe = self.pending_dir / "Dayflow_new.exe"
        if not new_exe.exists():
            logger.error("更新文件不存在")
            return False
        
        # 查找 updater.exe
        if getattr(sys, 'frozen', False):
            # 打包后的环境
            app_dir = Path(sys.executable).parent
        else:
            # 开发环境
            app_dir = Path(__file__).parent.parent
        
        updater_exe = app_dir / "updater.exe"
        
        # 如果 updater.exe 不存在，尝试使用 Python 脚本
        updater_script = app_dir / "updater.py"
        
        if updater_exe.exists():
            # 使用独立的 updater.exe
            subprocess.Popen(
                [str(updater_exe)],
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            return True
        elif updater_script.exists() and not getattr(sys, 'frozen', False):
            # 开发环境使用 Python 脚本
            subprocess.Popen(
                [sys.executable, str(updater_script)],
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            return True
        else:
            logger.error("找不到更新程序")
            return False
    
    def cleanup_pending_update(self):
        """清理待安装的更新"""
        if self.pending_dir.exists():
            shutil.rmtree(self.pending_dir, ignore_errors=True)
    
    @staticmethod
    def get_github_release_url() -> str:
        """获取 GitHub Release 页面 URL"""
        return f"https://github.com/{config.GITHUB_REPO}/releases"
    
    @staticmethod
    def get_mirror_release_url() -> str:
        """获取镜像 Release 页面 URL"""
        return f"https://mirror.ghproxy.com/https://github.com/{config.GITHUB_REPO}/releases"
