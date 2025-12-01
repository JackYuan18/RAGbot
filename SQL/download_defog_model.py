#!/usr/bin/env python3
"""
Utility script to download Defog SQLCoder model from HuggingFace.
Run this script to pre-download the model before using it in the application.
"""

import sys
import os
from pathlib import Path

def download_model(model_name: str = "defog/sqlcoder-7b-2"):
    """Download the Defog SQLCoder model from HuggingFace."""
    try:
        print(f"Downloading Defog SQLCoder model: {model_name}")
        print("This may take a while (model is ~14GB)...")
        print("=" * 60)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Download tokenizer
        print("\n1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("   [OK] Tokenizer downloaded successfully")
        
        # Download model
        print("\n2. Downloading model (this is the large file, ~14GB)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("   [OK] Model downloaded successfully")
        
        print("\n" + "=" * 60)
        print("[OK] Model download complete!")
        print(f"Model is cached at: {os.path.expanduser('~/.cache/huggingface/hub')}")
        print("You can now use Defog SQLCoder in the application.")
        
        return True
        
    except ImportError as e:
        print("\n[ERROR] Required packages not installed.")
        print("Please install transformers and torch:")
        print("  pip install transformers torch")
        return False
    except ConnectionError as e:
        print(f"\n[ERROR] Could not connect to HuggingFace.")
        print(f"Please check your internet connection.")
        print(f"Error: {str(e)}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "defog/sqlcoder-7b-2"
    success = download_model(model_name)
    sys.exit(0 if success else 1)

