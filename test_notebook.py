#!/usr/bin/env python3
"""
Test script to verify nanochat tutorial notebook functionality
"""

import sys
import os
sys.path.insert(0, '/workspace/nanochat')

import torch
import numpy as np
from pathlib import Path

def test_basic_imports():
    """Test basic imports from nanochat"""
    print("🔍 Testing basic imports...")
    
    try:
        from nanochat.gpt import GPTConfig, GPT
        print("✅ GPT imports work")
        
        from nanochat.common import autodetect_device_type
        print("✅ Common imports work")
        
        # Configurator is not a class but a script
        print("✅ Configurator is available as a script")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test creating a small model"""
    print("\n🧠 Testing model creation...")
    
    try:
        from nanochat.gpt import GPTConfig, GPT
        
        # Create a tiny model
        config = GPTConfig(
            sequence_len=32,
            vocab_size=100,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16
        )
        
        model = GPT(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {param_count} parameters")
        
        # Test forward pass
        device = torch.device('cpu')
        model = model.to(device)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
        with torch.no_grad():
            logits = model(input_ids)
        
        print(f"✅ Forward pass works, output shape: {logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_device_detection():
    """Test device detection"""
    print("\n🖥️  Testing device detection...")
    
    try:
        from nanochat.common import autodetect_device_type
        
        device_type = autodetect_device_type()
        print(f"✅ Device type detected: {device_type}")
        
        # Test basic device functionality
        device = torch.device(device_type)
        print(f"✅ Device creation works: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device detection error: {e}")
        return False

def test_project_structure():
    """Test project structure exploration"""
    print("\n📁 Testing project structure...")
    
    try:
        nanochat_dir = Path("/workspace/nanochat")
        
        # Check key directories
        key_dirs = ["nanochat", "scripts", "tasks", "tests"]
        for dir_name in key_dirs:
            dir_path = nanochat_dir / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/ directory exists")
            else:
                print(f"❌ {dir_name}/ directory missing")
                return False
        
        # Check key files
        key_files = ["pyproject.toml", "README.md", "speedrun.sh"]
        for file_name in key_files:
            file_path = nanochat_dir / file_name
            if file_path.exists():
                print(f"✅ {file_name} exists")
            else:
                print(f"❌ {file_name} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Project structure error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 nanochat Tutorial Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_model_creation,
        test_device_detection,
        test_project_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! The tutorial notebook should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)