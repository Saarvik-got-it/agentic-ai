#!/usr/bin/env python
"""
Setup script to validate RAG Agent installation and configuration.
Run this after installation to verify everything is working correctly.
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def check_python_version():
    """Verify Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        "langchain",
        "langchain_google_genai",
        "google.generativeai",
        "faiss",
        "dotenv",
        "pydantic"
    ]
    
    print("\nChecking dependencies...")
    all_ok = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_env_file():
    """Check if .env file exists and has API key."""
    print("\nChecking configuration...")
    
    if not Path(".env").exists():
        print("❌ .env file not found")
        print("   Create it with: cp .env.example .env")
        return False
    
    print("✅ .env file exists")
    
    # Check for API key
    with open(".env", "r") as f:
        content = f.read()
        if "GOOGLE_API_KEY" in content and "your_google_api_key_here" not in content:
            print("✅ GOOGLE_API_KEY configured")
            return True
        else:
            print("❌ GOOGLE_API_KEY not configured")
            print("   Edit .env and add your Google Gemini API key")
            return False


def check_directories():
    """Check if necessary directories exist."""
    print("\nChecking directory structure...")
    
    directories = [
        "agents",
        "utils",
        "pipelines",
        "data/documents",
        "vector_store",
        "logs"
    ]
    
    all_ok = True
    for directory in directories:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - MISSING")
            all_ok = False
    
    return all_ok


def check_imports():
    """Test importing main modules."""
    print("\nTesting module imports...")
    
    try:
        from utils.config import get_settings
        print("✅ Config module")
    except Exception as e:
        print(f"❌ Config module: {e}")
        return False
    
    try:
        from utils.logger import setup_logger
        print("✅ Logger module")
    except Exception as e:
        print(f"❌ Logger module: {e}")
        return False
    
    try:
        from agents.rag_agent import RAGAgent
        print("✅ RAG Agent module")
    except Exception as e:
        print(f"❌ RAG Agent module: {e}")
        return False
    
    return True


def check_documents():
    """Check if sample documents exist."""
    print("\nChecking documents...")
    
    doc_path = Path("data/documents")
    if not doc_path.exists():
        print("❌ data/documents/ directory not found")
        return False
    
    files = list(doc_path.glob("**/*.pdf")) + list(doc_path.glob("**/*.txt"))
    
    if files:
        print(f"✅ Found {len(files)} document(s):")
        for f in files:
            print(f"   - {f.name}")
        return True
    else:
        print("⚠️  No documents found in data/documents/")
        print("   Add PDF or TXT files to get started")
        return True  # Not an error, just a warning


def main():
    """Run all checks."""
    print("=" * 60)
    print("RAG Agent - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Configuration", check_env_file),
        ("Directory Structure", check_directories),
        ("Module Imports", check_imports),
        ("Documents", check_documents),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("✅ All checks passed! Ready to use RAG Agent.")
        print("\nNext steps:")
        print("1. Add your documents to: data/documents/")
        print("2. Run interactive mode: python main.py")
        print("3. Or ask a question: python main.py -q 'Your question'")
        return 0
    else:
        print(f"❌ {total - passed} check(s) failed. Fix errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
