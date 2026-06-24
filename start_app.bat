@echo off
REM Double-click to launch the local RAG web UI (http://127.0.0.1:7860).
REM Prerequisite: Ollama is running (tray icon) with qwen3:4b-instruct.
cd /d "%~dp0rag"
"E:\conda\envs\rag\python.exe" app.py
echo.
echo (App stopped. Close this window.)
pause
