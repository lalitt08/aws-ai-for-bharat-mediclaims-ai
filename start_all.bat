@echo off
title MediClaims AI - Unified Prototype

echo.
echo ================================================================
echo   MediClaims AI - Unified Healthcare Claims Prototype
echo ================================================================
echo.
echo Starting both systems:
echo   - Pre-Submission Pipeline  (http://localhost:5000)
echo   - Post-Submission Appeals  (http://localhost:8003)
echo.
echo Press Ctrl+C to stop everything.
echo ================================================================
echo.

cd /d "%~dp0"
python start_all.py

echo.
echo ================================================================
echo                    SYSTEM STOPPED
echo ================================================================
echo.
pause
