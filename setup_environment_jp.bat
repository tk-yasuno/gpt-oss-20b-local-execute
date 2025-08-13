@echo off
REM GPT-OSS B20 ローカル実行環境セットアップ
REM Python 3.12 + PyTorch CUDA環境

echo ========================================
echo GPT-OSS B20 Environment Setup
echo ========================================

REM 仮想環境の確認
if not exist "venv-gpt-oss\Scripts\activate.bat" (
    echo ❌ 仮想環境が見つかりません
    echo 📋 以下のコマンドで環境を作成してください:
    echo    py -3.12 -m venv venv-gpt-oss
    pause
    exit /b 1
)

REM 仮想環境をアクティベート
echo 🔧 仮想環境をアクティベート中...
call venv-gpt-oss\Scripts\activate.bat

REM Python環境確認
echo 📋 Python環境確認:
python --version
echo.

REM 必要なライブラリのインストール状況確認
echo 📦 インストール済みライブラリ確認:
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
echo 🎯 使用方法:
echo   基本テスト: python test_gpt_oss.py
echo   環境確認:   python -c "import torch; print(torch.cuda.is_available())"
echo ========================================
echo.

REM 自動でテストスクリプト実行するか確認
set /p choice="テストスクリプトを実行しますか？ (y/n): "
if /i "%choice%"=="y" (
    echo 🚀 GPT-OSS テスト実行中...
    python test_gpt_oss.py
) else (
    echo 💡 後でテストを実行する場合: python test_gpt_oss.py
)

pause
