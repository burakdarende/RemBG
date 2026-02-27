@echo off
title RemBG Pro — Arkaplan Silici
color 0A
cd /d "%~dp0"

echo.
echo  ============================================
echo    RemBG Pro — Gelismis Arkaplan Silici
echo         Neon / Tel Kafes / Fotograf
echo  ============================================
echo.

:: Venv varsa onu kullan, yoksa sistem Python
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    echo  [OK] Sanal ortam (venv) bulundu.
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo  [HATA] Python bulunamadi! python.org'dan yukleyin.
        pause
        exit /b 1
    )
    set PYTHON=python
    echo  [BILGI] Sistem Python kullaniliyor.
)

echo.

:: Gerekli paketler kurulu mu?
%PYTHON% -c "import numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo  [BILGI] Paketler yukleniyor...
    %PYTHON% -m pip install numpy pillow
)

echo  Uygulama baslatiliyor...
echo.
%PYTHON% rembg_app.py

if errorlevel 1 (
    echo.
    echo  [HATA] Uygulama hatayla kapandi.
    pause
)
