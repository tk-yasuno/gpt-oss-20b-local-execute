# GPT-OSS B20 ローカル実行環境

## 概要
GPT-OSS B20を**Python 3.12**環境でローカル実行するための環境です。

## 🎯 目的
- GPT-OSS B20モデルのローカル実行
- CUDA acceleration（RTX 4060 Ti対応）
- 高効率なテキスト生成処理

## 📋 システム要件
- **Python 3.12.x**
- **PyTorch 2.6.0+ (CUDA 12.4)**
- **GPU**: NVIDIA RTX 4060 Ti (15GB VRAM)
- **OS**: Windows 11

## 🚀 クイックスタート

### 1. 環境確認
```bash
setup_environment.bat
```

### 2. テスト実行
```bash
# 基本テスト
python test_gpt_oss.py

# または仮想環境から
venv-gpt-oss\Scripts\python.exe test_gpt_oss.py
```

## 📁 ファイル構成
```
gpt-oss-20b/
├── venv-gpt-oss/          # Python 3.12仮想環境
├── test_gpt_oss.py        # 包括的テストスクリプト
├── config.py              # 設定ファイル
├── setup_environment.bat  # 環境セットアップ
└── README.md              # このファイル
```

## 🔧 インストール済みライブラリ
- **PyTorch** 2.6.0+cu124
- **Transformers** 4.55.1
- **Accelerate** 1.2.1
- **bitsandbytes** 0.45.0
- **datasets** 3.2.0

## ⚡ 性能最適化
- CUDA acceleration対応
- 8bit/4bit量子化オプション
- バッチ処理最適化
- メモリ効率的な実行

## 🛠️ トラブルシューティング

### CUDA関連
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

### メモリ不足時
- `config.py`で`USE_8BIT = True`を設定
- `GPU_MEMORY_FRACTION`を調整

## 📊 テスト項目
1. ✅ Python 3.12環境確認
2. ✅ PyTorch CUDA動作確認
3. ✅ GPU メモリチェック
4. ✅ モデル読み込みテスト
5. ✅ テキスト生成テスト
6. ✅ 性能ベンチマーク

## 📝 使用例
```python
from test_gpt_oss import load_gpt_oss_model, test_text_generation

# モデル読み込み
model, tokenizer = load_gpt_oss_model()

# テキスト生成
result = test_text_generation(model, tokenizer, "こんにちは、")
print(result)
```

---
**Created**: 2024年12月28日  
**Environment**: Python 3.12.10 + PyTorch 2.6.0+cu124  
**Status**: ✅ Ready for GPT-OSS B20 testing
