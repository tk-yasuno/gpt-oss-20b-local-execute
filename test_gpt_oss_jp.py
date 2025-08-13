#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-OSS B20 ローカル実行テスト
================================

Python 3.12環境でgpt-oss-B20モデルのローカル実行をテストするスクリプト

環境要件:
- Python 3.12.10
- PyTorch 2.6.0+cu124  
- Transformers 4.55.1
- CUDA 12.4対応GPU (RTX 4060 Ti推奨)
"""

import torch
import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """システム要件の確認"""
    logger.info("🔍 システム要件チェック開始")
    
    # Python バージョン
    import sys
    python_version = sys.version
    logger.info(f"🐍 Python: {python_version.split()[0]}")
    
    # PyTorch & CUDA
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"⚡ CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        logger.info(f"🚀 GPU: {gpu_name}")
        logger.info(f"💾 GPU Memory: {gpu_memory}GB")
    else:
        logger.warning("⚠️ CUDA not available - CPU inference only")
    
    # システムRAM
    ram_total = psutil.virtual_memory().total // (1024**3)
    ram_available = psutil.virtual_memory().available // (1024**3)
    logger.info(f"🧠 System RAM: {ram_available}GB/{ram_total}GB available")
    
    return torch.cuda.is_available()

def load_gpt_oss_model(model_name="microsoft/DialoGPT-medium", use_cuda=True):
    """
    GPT-OSSモデルのロード (テスト用にDialoGPTを使用)
    実際のgpt-oss-B20が利用可能になったら、model_nameを変更
    """
    logger.info(f"📥 モデル読み込み開始: {model_name}")
    
    try:
        # デバイス設定
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Device: {device}")
        
        # トークナイザーとモデルのロード
        logger.info("📋 Tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("🤖 Model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
            
        logger.info("✅ Model loaded successfully!")
        
        # メモリ使用量確認
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() // (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            logger.info(f"💾 GPU Memory Used: {gpu_memory_used}GB/{gpu_memory_total}GB")
        
        return tokenizer, model, device
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return None, None, None

def test_text_generation(tokenizer, model, device, prompt="こんにちは！今日はどんな日ですか？"):
    """テキスト生成テスト"""
    logger.info("🎯 テキスト生成テスト開始")
    logger.info(f"📝 Prompt: {prompt}")
    
    try:
        # 入力エンコード
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # 推論実行
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,  # 50トークン追加生成
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        generation_time = time.time() - start_time
        
        # 結果デコード
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"📄 Generated text: {generated_text}")
        logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
        logger.info(f"📊 Tokens/second: {50/generation_time:.2f}")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"❌ Text generation failed: {e}")
        return None

def benchmark_performance(tokenizer, model, device, num_tests=3):
    """パフォーマンスベンチマーク"""
    logger.info(f"🏁 Performance benchmark ({num_tests} tests)")
    
    test_prompts = [
        "人工知能の未来について教えてください。",
        "今日の天気はどうですか？",
        "おすすめの本を教えてください。"
    ]
    
    total_times = []
    
    for i, prompt in enumerate(test_prompts[:num_tests]):
        logger.info(f"📋 Test {i+1}/{num_tests}: {prompt[:30]}...")
        
        start_time = time.time()
        result = test_text_generation(tokenizer, model, device, prompt)
        test_time = time.time() - start_time
        
        total_times.append(test_time)
        
        if result:
            logger.info(f"✅ Test {i+1} completed in {test_time:.2f}s")
        else:
            logger.error(f"❌ Test {i+1} failed")
    
    if total_times:
        avg_time = sum(total_times) / len(total_times)
        logger.info(f"📊 Average generation time: {avg_time:.2f}s")
        logger.info(f"🚀 Performance: {50/avg_time:.2f} tokens/second")

def main():
    """メイン実行関数"""
    logger.info("🚀 GPT-OSS B20 ローカル実行テスト開始")
    logger.info("=" * 60)
    
    # システム要件チェック
    cuda_available = check_system_requirements()
    logger.info("=" * 60)
    
    # モデルロード
    tokenizer, model, device = load_gpt_oss_model(use_cuda=cuda_available)
    
    if tokenizer is None or model is None:
        logger.error("❌ Model loading failed. Exiting.")
        return
    
    logger.info("=" * 60)
    
    # 基本テスト
    test_text_generation(tokenizer, model, device)
    logger.info("=" * 60)
    
    # ベンチマーク
    benchmark_performance(tokenizer, model, device)
    logger.info("=" * 60)
    
    logger.info("🎉 GPT-OSS B20 テスト完了！")
    
    # メモリクリーンアップ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 GPU memory cleared")

if __name__ == "__main__":
    main()
