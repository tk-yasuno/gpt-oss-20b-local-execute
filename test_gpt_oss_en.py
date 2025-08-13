#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-OSS B20 Local Execution Test
=================================

Script for testing local execution of gpt-oss-B20 model in Python 3.12 environment

Environment Requirements:
- Python 3.12.10
- PyTorch 2.6.0+cu124  
- Transformers 4.55.1
- CUDA 12.4 compatible GPU (RTX 4060 Ti recommended)
"""

import torch
import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements"""
    logger.info("🔍 System requirements check started")
    
    # Python version
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
    
    # System RAM
    ram_total = psutil.virtual_memory().total // (1024**3)
    ram_available = psutil.virtual_memory().available // (1024**3)
    logger.info(f"🧠 System RAM: {ram_available}GB/{ram_total}GB available")
    
    return torch.cuda.is_available()

def load_gpt_oss_model(model_name="microsoft/DialoGPT-medium", use_cuda=True):
    """
    Load GPT-OSS model (using DialoGPT for testing)
    Change model_name when actual gpt-oss-B20 becomes available
    """
    logger.info(f"📥 Model loading started: {model_name}")
    
    try:
        # Device configuration
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Device: {device}")
        
        # Load tokenizer and model
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
        
        # Check memory usage
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() // (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            logger.info(f"💾 GPU Memory Used: {gpu_memory_used}GB/{gpu_memory_total}GB")
        
        return tokenizer, model, device
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return None, None, None

def test_text_generation(tokenizer, model, device, prompt="Hello! How are you doing today?"):
    """Text generation test"""
    logger.info("🎯 Text generation test started")
    logger.info(f"📝 Prompt: {prompt}")
    
    try:
        # Input encoding
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,  # Generate 50 additional tokens
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        generation_time = time.time() - start_time
        
        # Decode results
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"📄 Generated text: {generated_text}")
        logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
        logger.info(f"📊 Tokens/second: {50/generation_time:.2f}")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"❌ Text generation failed: {e}")
        return None

def benchmark_performance(tokenizer, model, device, num_tests=3):
    """Performance benchmark"""
    logger.info(f"🏁 Performance benchmark ({num_tests} tests)")
    
    test_prompts = [
        "Please tell me about the future of artificial intelligence.",
        "What's the weather like today?",
        "Can you recommend a good book to read?"
    ]
    
    total_times = []
    
    for i, prompt in enumerate(test_prompts[:num_tests]):
        logger.info(f"📋 Test {i+1}/{num_tests}: {prompt[:50]}...")
        
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
    """Main execution function"""
    logger.info("🚀 GPT-OSS B20 Local Execution Test Started")
    logger.info("=" * 60)
    
    # System requirements check
    cuda_available = check_system_requirements()
    logger.info("=" * 60)
    
    # Model loading
    tokenizer, model, device = load_gpt_oss_model(use_cuda=cuda_available)
    
    if tokenizer is None or model is None:
        logger.error("❌ Model loading failed. Exiting.")
        return
    
    logger.info("=" * 60)
    
    # Basic test
    test_text_generation(tokenizer, model, device)
    logger.info("=" * 60)
    
    # Benchmark
    benchmark_performance(tokenizer, model, device)
    logger.info("=" * 60)
    
    logger.info("🎉 GPT-OSS B20 Test Completed!")
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 GPU memory cleared")

if __name__ == "__main__":
    main()
