@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-FirefoxDevelopment.ps1" %*
if errorlevel 1 (
  echo.
  echo Hskify development mode could not start. See the error above.
  pause
)
