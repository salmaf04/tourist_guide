#!/usr/bin/env python3
"""
Script to install required dependencies for the RAG system
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install all required packages"""
    print("Installing dependencies for Tourist Guide RAG system...")
    
    # Required packages
    packages = [
        "streamlit>=1.28.0",
        "sentence-transformers>=2.2.2", 
        "geopy>=2.3.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "python-dotenv>=1.0.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually using:")
        for package in failed_packages:
            print(f"  pip install {package}")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)