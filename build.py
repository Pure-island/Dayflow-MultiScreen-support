"""
Dayflow 打包脚本
使用 PyInstaller 将应用打包为独立 EXE
"""

import subprocess
import sys


def build():
    """打包应用"""

    # PyInstaller 参数
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=Dayflow",
        "--onedir",  # 生成目录
        "--windowed",  # 无控制台窗口
        "--clean",  # 清理缓存
        "--icon=assets/icon.ico",  # 应用图标
        "--add-data=database/schema.sql;database",  # 包含数据库架构
        "--add-data=templates;templates",  # 包含 HTML 模板
        # PySide6 6.8.1 通常只需要这些
        "--hidden-import=PySide6.QtSvg",
        "--hidden-import=PySide6.QtSvgWidgets",
        "--collect-all=dxcam",  # 收集 dxcam 所有文件
        "--collect-all=rapidocr_onnxruntime",  # OCR 依赖
        "--collect-all=onnxruntime",  # OCR 运行时
        "--noconfirm",  # 覆盖已有输出
        "main.py",
    ]

    print("=" * 50)
    print("  Dayflow 打包工具")
    print("=" * 50)
    print()
    print("正在打包，请稍候...")
    print()

    try:
        subprocess.run(args, check=True)
        print()
        print("=" * 50)
        print("  ✅ 打包成功！")
        print("  输出目录: dist/Dayflow/")
        print("  运行: dist/Dayflow/Dayflow.exe")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"❌ 打包失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 请先安装 PyInstaller:")
        print("   pip install pyinstaller")
        sys.exit(1)


if __name__ == "__main__":
    build()
