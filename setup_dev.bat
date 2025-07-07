@echo off
echo 🚀 Setting up AI Customer Intelligence Agent
echo ============================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo ⚡ Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating project directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\exports 2>nul
mkdir logs 2>nul
mkdir static\css 2>nul
mkdir static\js 2>nul
mkdir static\images 2>nul

REM Check if .env exists
if not exist ".env" (
    echo 🔑 Please create .env file and add your OpenAI API key
    echo ⚠️  Copy .env.example to .env and edit it
)

echo.
echo 🎉 Setup complete!
echo.
echo 📋 Next steps:
echo 1. Edit .env file and add your OpenAI API key
echo 2. Run: streamlit run app.py
echo 3. Open your browser to http://localhost:8501
echo.
pause
