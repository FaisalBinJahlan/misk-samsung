@echo off
cd /d "%~dp0"
python -m http.server 3000 --bind 127.0.0.1
if errorlevel 1 pause
