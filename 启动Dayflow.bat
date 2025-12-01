@echo off
chcp 65001 >nul
title Dayflow - 智能时间追踪

echo ========================================
echo   Dayflow for Windows
echo   正在启动...
echo ========================================
echo.

:: 切换到脚本所在目录
cd /d "D:\github\Dayflow"

:: 使用 conda run 直接运行
conda run -n dayflow --no-capture-output python main.py

:: 如果程序异常退出，暂停查看错误
if errorlevel 1 (
    echo.
    echo [程序异常退出]
    echo 请确保已安装依赖: pip install -r requirements.txt
    pause
)
