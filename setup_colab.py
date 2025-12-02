"""
setup script for google colab
run this first to install dependencies and prepare the environment
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    """run a shell command"""
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0

def main():
    print("="*60)
    print("setting up nanochat for colab")
    print("="*60)
    
    # install dependencies
    print("\n📦 installing dependencies...")
    deps = [
        "torch>=2.0.0",
        "tiktoken>=0.5.0",
        "numpy>=1.24.0",
    ]
    
    for dep in deps:
        if not run_cmd(f"{sys.executable} -m pip install -q {dep}"):
            print(f"⚠️  failed to install {dep}")
    
    print("\n✅ setup complete!")
    print("\nnext steps:")
    print("1. upload your training_data/chatml_format.txt to colab")
    print("2. run: python -m nanochat.train")
    print("\nor use the train_colab.py script for a complete workflow")

if __name__ == "__main__":
    main()

