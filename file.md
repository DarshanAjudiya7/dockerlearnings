# Detect environment
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab - Perfect!")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally - Nice!")

if IN_COLAB:
    print("\n📦 Cloning OpenEnv repository...")
    !git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1
    %cd OpenEnv
    
    print("📚 Installing dependencies (this takes ~10 seconds)...")
    !pip install -q fastapi uvicorn requests
    
    import sys
    sys.path.insert(0, './src')
    print("\n✅ Setup complete! Everything is ready to go! 🎉")
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent / 'src'))
    print("✅ Using local OpenEnv installation")

print("\n🚀 Ready to explore OpenEnv and build amazing things!")
print("💡 Tip: Run cells top-to-bottom for the best experience.\n")