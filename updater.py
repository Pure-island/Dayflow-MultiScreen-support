"""
Dayflow Updater - 独立更新程序
职责：等待主程序退出 → 替换 EXE → 重启主程序

此脚本需要单独打包为 updater.exe：
    pyinstaller --onefile --noconsole --name updater updater.py
"""
import os
import sys
import time
import json
import shutil
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
log_dir = Path(os.getenv('LOCALAPPDATA', '')) / 'Dayflow' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'updater_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_app_data_dir() -> Path:
    """获取应用数据目录"""
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "Dayflow"
    user_profile = os.getenv("USERPROFILE")
    if user_profile:
        return Path(user_profile) / "AppData" / "Local" / "Dayflow"
    return Path.home() / "AppData" / "Local" / "Dayflow"


def wait_for_process_exit(exe_path: Path, timeout: int = 30) -> bool:
    """
    等待指定进程退出
    
    通过尝试重命名文件来检测文件是否被占用
    """
    logger.info(f"等待主程序退出: {exe_path}")
    
    for i in range(timeout):
        try:
            # 尝试重命名来检测文件是否被占用
            temp_name = exe_path.with_suffix('.exe.tmp_check')
            exe_path.rename(temp_name)
            temp_name.rename(exe_path)
            logger.info("主程序已退出")
            return True
        except PermissionError:
            logger.info(f"等待中... ({i+1}/{timeout})")
            time.sleep(1)
        except FileNotFoundError:
            # 文件不存在，可能是首次安装
            logger.info("原文件不存在，跳过等待")
            return True
    
    logger.error("等待超时，主程序可能仍在运行")
    return False


def apply_update() -> bool:
    """执行更新"""
    app_data_dir = get_app_data_dir()
    pending_dir = app_data_dir / "pending_update"
    info_path = pending_dir / "update_info.json"
    new_exe = pending_dir / "Dayflow_new.exe"
    
    # 读取更新信息
    if not info_path.exists():
        logger.error("找不到更新信息文件")
        return False
    
    try:
        info = json.loads(info_path.read_text())
    except Exception as e:
        logger.error(f"读取更新信息失败: {e}")
        return False
    
    if not info.get('ready'):
        logger.error("更新未准备就绪")
        return False
    
    # 获取原 EXE 路径
    current_exe_path = info.get('current_exe_path', '')
    if not current_exe_path:
        logger.error("未找到原 EXE 路径")
        return False
    
    current_exe = Path(current_exe_path)
    app_dir = current_exe.parent  # 应用目录（包含 DLL 等）
    backup_dir = app_data_dir / "backup"
    
    logger.info(f"原 EXE: {current_exe}")
    logger.info(f"应用目录: {app_dir}")
    logger.info(f"新 EXE: {new_exe}")
    
    # 检查新版本文件
    if not new_exe.exists():
        logger.error("新版本文件不存在")
        return False
    
    # 等待主程序退出
    if current_exe.exists():
        if not wait_for_process_exit(current_exe):
            logger.warning("无法确认主程序已退出，尝试继续...")
    
    try:
        # 备份旧版本 EXE
        logger.info("备份旧版本...")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_exe = backup_dir / "Dayflow.exe.backup"
        if current_exe.exists():
            if backup_exe.exists():
                backup_exe.unlink()
            shutil.copy2(current_exe, backup_exe)
        
        # 替换 EXE
        logger.info("安装新版本 EXE...")
        if current_exe.exists():
            current_exe.unlink()
        shutil.copy2(new_exe, current_exe)
        
        # 复制其他文件（DLL、_internal 目录等）
        logger.info("复制依赖文件...")
        skip_files = {'Dayflow_new.exe', 'update_info.json'}
        skip_extensions = {'.tmp', '.zip'}
        
        for item in pending_dir.iterdir():
            if item.name in skip_files:
                continue
            if item.suffix.lower() in skip_extensions:
                continue
            
            dest = app_dir / item.name
            
            if item.is_file():
                logger.info(f"  复制文件: {item.name}")
                if dest.exists():
                    dest.unlink()
                shutil.copy2(item, dest)
            elif item.is_dir():
                logger.info(f"  复制目录: {item.name}")
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
        
        # 清理待更新目录
        logger.info("清理临时文件...")
        shutil.rmtree(pending_dir, ignore_errors=True)
        
        logger.info("更新完成!")
        return True
        
    except Exception as e:
        logger.error(f"更新失败: {e}")
        
        # 尝试回滚
        backup_exe = backup_dir / "Dayflow.exe.backup"
        if backup_exe.exists():
            logger.info("尝试回滚...")
            try:
                if current_exe.exists():
                    current_exe.unlink()
                shutil.copy2(backup_exe, current_exe)
                logger.info("回滚成功")
            except Exception as rollback_error:
                logger.error(f"回滚失败: {rollback_error}")
        
        return False


def restart_app(exe_path: str):
    """重启主程序"""
    logger.info(f"启动主程序: {exe_path}")
    
    try:
        if os.path.exists(exe_path):
            subprocess.Popen(
                [exe_path],
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            logger.info("主程序已启动")
        else:
            logger.error(f"主程序不存在: {exe_path}")
    except Exception as e:
        logger.error(f"启动主程序失败: {e}")


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("Dayflow Updater 启动")
    logger.info("=" * 50)
    
    # 短暂延迟，确保主程序有时间退出
    time.sleep(1)
    
    # 读取更新信息获取原 EXE 路径
    app_data_dir = get_app_data_dir()
    info_path = app_data_dir / "pending_update" / "update_info.json"
    
    current_exe_path = ""
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            current_exe_path = info.get('current_exe_path', '')
        except:
            pass
    
    # 执行更新
    success = apply_update()
    
    # 重启主程序
    if current_exe_path:
        restart_app(current_exe_path)
    else:
        logger.warning("无法获取主程序路径，请手动启动")
    
    logger.info("Updater 退出")
    
    # 给一点时间让日志写入
    time.sleep(0.5)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"Updater 发生未处理异常: {e}")
        sys.exit(1)
