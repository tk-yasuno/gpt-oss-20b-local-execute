# GPT-OSS B20 Configuration
# ローカル実行用設定

# モデル関連設定
MODEL_NAME = "cyberagent/open-calm-7b"  # 代替モデル例
MODEL_CACHE_DIR = "./models"
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# GPUメモリ関連
GPU_MEMORY_FRACTION = 0.8  # GPU使用率を80%に制限
USE_CUDA = True
DEVICE_MAP = "auto"

# 8bit/4bit量子化オプション
USE_8BIT = False  # メモリ節約したい場合はTrue
USE_4BIT = False  # さらにメモリ節約したい場合はTrue

# バッチ処理設定
BATCH_SIZE = 1
NUM_BEAMS = 1  # ビームサーチ無効（高速化）

# ロギング設定
LOG_LEVEL = "INFO"
VERBOSE = True

# テスト用プロンプト
TEST_PROMPTS = [
    "こんにちは、今日の天気は",
    "AIについて簡単に説明すると、",
    "プログラミングを学ぶ理由は",
    "日本の文化の特徴として"
]

# システム要件
MIN_PYTHON_VERSION = (3, 12)
MIN_PYTORCH_VERSION = "2.6.0"
REQUIRED_GPU_MEMORY = 4  # GB
RECOMMENDED_GPU_MEMORY = 8  # GB
