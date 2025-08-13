@echo off
REM GPT-OSS B20 Local Execution Environment Setup
REM Python 3.12 + PyTorch CUDA Environment

echo ========================================
echo GPT-OSS B20 Environment Setup
echo ========================================

REM Check virtual environment
if not exist "venv-gpt-oss\Scripts\activate.bat" (
    echo ❌ Virtual environment not found
    echo 📋 Please create environment with:
    echo    py -3.12 -m venv venv-gpt-oss
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv-gpt-oss\Scripts\activate.bat

REM Check Python environment
echo 📋 Python environment check:
python --version
echo.

REM Check installed libraries
echo 📦 Checking installed libraries:
python -c "
try:
    import torch; print('✅ PyTorch:', torch.__version__)
except: print('❌ PyTorch not installed')

try:
    import transformers; print('✅ Transformers:', transformers.__version__)
except: print('❌ Transformers not installed')

try:
    import accelerate; print('✅ Accelerate:', accelerate.__version__)
except: print('❌ Accelerate not installed')

print('🚀 CUDA Available:', torch.cuda.is_available() if 'torch' in locals() else 'Unknown')
"

echo.
echo ========================================
echo 🎯 Usage:
echo   Basic test: python test_gpt_oss_en.py
echo   Environment check: python -c "import torch; print(torch.cuda.is_available())"
echo ========================================
echo.

REM Ask to run test script automatically
set /p choice="Run test script now? (y/n): "
if /i "%choice%"=="y" (
    echo 🚀 Running GPT-OSS test...
    python test_gpt_oss_en.py
) else (
    echo 💡 To run test later: python test_gpt_oss_en.py
)

pause
