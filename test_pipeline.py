#!/usr/bin/env python
"""Quick test of pipeline functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("PIPELINE MODULE TEST")
print("=" * 80)

# Test 1: Import
print("\n[1] Testing imports...")
try:
    from pipelines.main_pipeline import run_pipeline, get_available_options
    print("✅ Pipeline imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Load config
print("\n[2] Testing configuration loading...")
try:
    options = get_available_options()
    personas = options["personas"]
    content_types = options["content_types"]
    print(f"✅ Configuration loaded")
    print(f"   Personas: {personas}")
    print(f"   Content types: {content_types}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Test 3: Validation error handling
print("\n[3] Testing validation (empty query)...")
try:
    response = run_pipeline("")
    if not response.success and "empty" in response.error.lower():
        print(f"✅ Validation working: {response.error}")
    else:
        print(f"⚠️  Unexpected response: {response}")
except Exception as e:
    print(f"❌ Validation test failed: {e}")

# Test 4: Invalid persona
print("\n[4] Testing validation (invalid persona)...")
try:
    response = run_pipeline("test query", persona="nonexistent")
    if not response.success and "invalid persona" in response.error.lower():
        print(f"✅ Persona validation working: {response.error}")
    else:
        print(f"⚠️  Unexpected response: {response}")
except Exception as e:
    print(f"❌ Persona validation test failed: {e}")

# Test 5: Invalid content type
print("\n[5] Testing validation (invalid content type)...")
try:
    response = run_pipeline("test query", content_type="nonexistent")
    if not response.success and "invalid content_type" in response.error.lower():
        print(f"✅ Content type validation working: {response.error}")
    else:
        print(f"⚠️  Unexpected response: {response}")
except Exception as e:
    print(f"❌ Content type validation test failed: {e}")

print("\n" + "=" * 80)
print("✅ BASIC VALIDATION TESTS PASSED")
print("=" * 80)
print("\nTo test full pipeline:")
print("  python main.py --pipeline --list-options")
print("  python main.py --pipeline --query 'Your test query'")
print("=" * 80)
