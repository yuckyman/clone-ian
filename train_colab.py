"""
colab-friendly training script
handles setup, data upload, and training in one go
"""

import os
import sys
from pathlib import Path

# add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def setup_colab():
    """setup colab environment"""
    print("🔧 setting up colab environment...")
    
    # install dependencies
    import subprocess
    deps = ["torch>=2.0.0", "tiktoken>=0.5.0", "numpy>=1.24.0"]
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], check=False)
    
    print("✅ dependencies installed")

def check_data():
    """check if training data exists"""
    data_path = Path("training_data/chatml_format.txt")
    if not data_path.exists():
        print("\n⚠️  training data not found!")
        print(f"expected: {data_path}")
        print("\noptions:")
        print("1. upload training_data/chatml_format.txt to colab")
        print("2. or mount google drive and place it there")
        print("3. or clone your repo with the data")
        return False
    return True

def main():
    # detect colab
    in_colab = 'google.colab' in sys.modules or os.path.exists('/content')
    
    if in_colab:
        print("🤖 detected google colab")
        setup_colab()
    
    # check data
    if not check_data():
        print("\n❌ cannot proceed without training data")
        return
    
    # import and run training
    print("\n🚀 starting training...")
    from nanochat.train import main as train_main
    train_main()

if __name__ == "__main__":
    main()

