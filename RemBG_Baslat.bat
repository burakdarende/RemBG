@echo off
setlocal EnableDelayedExpansion
title RemBG Pro

:: Bat'in bulundugu klasore git (her durumda calisir)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo  ============================================
echo    RemBG Pro -- Gelismis Arkaplan Silici
echo  ============================================
echo.
echo  Klasor: %SCRIPT_DIR%
echo.

:: Venv'i kontrol et
set "VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    echo  [OK] Sanal ortam bulundu.
    set "PYTHON=%VENV_PYTHON%"
) else (
    echo  [BILGI] Venv bulunamadi, sistem Python kullaniliyor...
    where python >nul 2>&1
    if errorlevel 1 (
        echo  [HATA] Python bulunamadi!
        echo  Lutfen python.org adresinden Python yukleyin.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

echo  Python: !PYTHON!
echo.

:: Bagimlilik kontrolu
"%PYTHON%" -c "import numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo  [BILGI] Eksik paket kuruluyor...
    "%PYTHON%" -m pip install numpy pillow
    echo.
)

:: Uygulamayi baslat
echo  Uygulama baslatiliyor...
echo.
"%PYTHON%" "%SCRIPT_DIR%rembg_app.py"

if errorlevel 1 (
    echo.
    echo  [HATA] Uygulama hatayla kapandi!
    echo  Hata kodu: %errorlevel%
    pause
)
endlocal
