#!/usr/bin/env python3
"""
Test script for 30-day readmission evaluation with lighteval.
"""

import os
import sys

def test_config_validation():
    """Test that the configuration can be loaded and validated."""
    try:
        from clinicaleval.cli import load_config
        
        # Test loading the report-distillation config
        cfg = load_config('configs/report-distillation.yaml', [])
        
        print("✓ Configuration loaded successfully")
        
        # Check required fields
        assert 'data' in cfg, "Missing data configuration"
        assert 'lighteval' in cfg, "Missing lighteval configuration" 
        assert 'report' in cfg, "Missing report configuration"
        
        print("✓ Required configuration sections present")
        
        # Test with lighteval enabled
        overrides = [
            "lighteval.enabled=true",
            "lighteval.model_path=/ssd-shared/qwen/Qwen3-1.7B",
        ]
        cfg_enabled = load_config('configs/report-distillation.yaml', overrides)
        
        assert cfg_enabled['lighteval']['enabled'] is True
        assert cfg_enabled['lighteval']['model_path'] == "/ssd-shared/qwen/Qwen3-1.7B"
        
        print("✓ Configuration overrides working")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_custom_task_loading():
    """Test that the custom task can be imported."""
    try:
        from clinicaleval.custom_tasks import TASKS_TABLE, readmission_task
        
        assert len(TASKS_TABLE) > 0, "No tasks registered"
        assert readmission_task.name == "readmission_30day"
        
        print("✓ Custom tasks loaded successfully")
        print(f"  - Task name: {readmission_task.name}")
        print(f"  - Suite: {readmission_task.suite}")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom task test failed: {e}")
        return False


def test_data_loading():
    """Test that data can be loaded from the configured path."""
    try:
        from clinicaleval.cli import read_json, read_jsonl
        
        # Test with sample data first
        sample_data = read_jsonl('data/sample.jsonl', 5)
        assert len(sample_data) > 0, "No sample data loaded"
        
        print("✓ Sample data loading works")
        
        # Check if the actual dataset exists
        dataset_path = "/ssd-shared/report-distillation.json"
        if os.path.exists(dataset_path):
            data = read_json(dataset_path, 5)
            print(f"✓ Found {len(data)} records in {dataset_path}")
            
            # Check data structure
            if data:
                sample = data[0]
                required_keys = ['text', 'label']
                for key in required_keys:
                    if key not in sample:
                        print(f"⚠ Warning: '{key}' not found in data. Available keys: {list(sample.keys())}")
                        
        else:
            print(f"⚠ Warning: Dataset not found at {dataset_path}")
            
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing 30-day readmission evaluation setup...")
    print("=" * 50)
    
    tests = [
        test_config_validation,
        test_custom_task_loading,
        test_data_loading,
    ]
    
    passed = 0
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nTo run the evaluation with lighteval:")
        print("clinicaleval \\")
        print("  --config configs/report-distillation.yaml \\")
        print("  lighteval.enabled=true \\")
        print("  lighteval.model_path=/ssd-shared/qwen/Qwen3-1.7B")
    else:
        print(f"\n❌ {len(tests) - passed} test(s) failed. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()