@echo off
cd /d "%~dp0"

REM Intenta usar py (launcher de Python) primero, sino busca python en PATH
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 transcribir_carpeta.py
) else (
    python transcribir_carpeta.py
)

echo.
pause
