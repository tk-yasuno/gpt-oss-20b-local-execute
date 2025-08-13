# GPT-OSS B20 Configuration
# Configuration for local execution

# Model related settings
MODEL_NAME = "cyberagent/open-calm-7b"  # Alternative model example
MODEL_CACHE_DIR = "./models"
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# GPU memory settings
GPU_MEMORY_FRACTION = 0.8  # Limit GPU usage to 80%
USE_CUDA = True
DEVICE_MAP = "auto"

# 8bit/4bit quantization options
USE_8BIT = False  # Set to True for memory saving
USE_4BIT = False  # Set to True for further memory saving

# Batch processing settings
BATCH_SIZE = 1
NUM_BEAMS = 1  # Disable beam search (for speed)

# Logging settings
LOG_LEVEL = "INFO"
VERBOSE = True

# Test prompts
TEST_PROMPTS = [
    "Hello, how's the weather today?",
    "To explain AI in simple terms,",
    "The reason to learn programming is",
    "The characteristics of American culture include"
]

# System requirements
MIN_PYTHON_VERSION = (3, 12)
MIN_PYTORCH_VERSION = "2.6.0"
REQUIRED_GPU_MEMORY = 4  # GB
RECOMMENDED_GPU_MEMORY = 8  # GB
