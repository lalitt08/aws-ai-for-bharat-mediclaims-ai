@echo off
title Healthcare Claims AI - Agentic System

echo.
echo ================================================================
echo             HEALTHCARE CLAIMS AI - AGENTIC SYSTEM
echo ================================================================
echo.
echo Starting the complete agentic system...
echo.
echo This will start:
echo   - MCP Server (AI tools)
echo   - Primary Insurance API (BlueCross/Aetna)
echo   - Secondary Insurance API (Cigna/United)
echo   - Agent Orchestrator (multi-agent coordination)
echo   - Web Dashboard (user interface)
echo.
echo Access the system at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the system
echo.
echo ================================================================
echo.

cd /d "%~dp0"
python start_agentic_system.py

echo.
echo ================================================================
echo                    SYSTEM STOPPED
echo ================================================================
echo.
pause
