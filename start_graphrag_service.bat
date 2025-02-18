@echo off
echo Starting GraphRAG Service...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)

REM Check if required packages are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing FastAPI...
    pip install fastapi uvicorn
)

pip show fast-graphrag >nul 2>&1
if errorlevel 1 (
    echo Installing fast-graphrag...
    pip install fast-graphrag
)

pip show sentence-transformers >nul 2>&1
if errorlevel 1 (
    echo Installing sentence-transformers...
    pip install sentence-transformers
)

pip show google-generativeai >nul 2>&1
if errorlevel 1 (
    echo Installing Google Generative AI...
    pip install google-generativeai
)

REM Start the service
echo Starting GraphRAG service on http://localhost:8000
python graphrag_service.py 