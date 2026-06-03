#!/usr/bin/env python3
"""
Test script to verify Self-CoT functionality.
This script will test the Self-CoT data loading and training mode switching.
"""

import os
import sys
import json
import tempfile
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset_utils.gsm8k import GSM8KDataset, DatasetState


def create_test_self_cot_data():
    """Create a small test Self-CoT data file."""
    test_data = [
        {
            "question": "Janet's dogs eat 2 pounds of food each week. How many pounds of food do her dogs eat in 8 weeks?",
            "gt_answer": "Janet's dogs eat 2 pounds of food each week.\nTo find out how many pounds they eat in 8 weeks, we multiply the weekly amount by the number of weeks.\n2 pounds/week × 8 weeks = 16 pounds\n#### 16",
            "gt_numerical": "16",
            "self_cot": "Let me solve this step by step:\n\n1) Janet's dogs eat 2 pounds of food each week\n2) We need to find out how much they eat in 8 weeks\n3) To do this, we multiply the weekly amount by the number of weeks\n4) 2 pounds × 8 weeks = 16 pounds\n\nTherefore, Janet's dogs eat \\boxed{16} pounds of food in 8 weeks.",
            "predicted_answer": "16",
            "is_correct": True
        },
        {
            "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees in the grove. How many trees did the grove workers plant today?",
            "gt_answer": "There are 15 trees in the grove.\nAfter the workers plant trees, there will be 21 trees.\nTo find how many trees were planted, we subtract the original number from the final number.\n21 - 15 = 6\n#### 6",
            "gt_numerical": "6",
            "self_cot": "Let me think through this:\n\n1) Initially there are 15 trees in the grove\n2) After planting, there will be 21 trees\n3) To find how many were planted, I need to find the difference\n4) 21 - 15 = 6\n\nSo the grove workers planted \\boxed{6} trees today.",
            "predicted_answer": "6",
            "is_correct": True
        },
        {
            "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "gt_answer": "Leah had 32 chocolates and her sister had 42.\nSo in total they had 32 + 42 = 74 chocolates.\nIf they ate 35, they have 74 - 35 = 39 chocolates left.\n#### 39",
            "gt_numerical": "39",
            "self_cot": "I made a mistake in my calculation.\n\n1) Leah has 32 chocolates\n2) Her sister has 42 chocolates\n3) Total: 32 + 42 = 74\n4) They ate 35\n5) Remaining: 74 - 35 = 39\n\nWait, let me recalculate... 32 + 42 = 74, 74 - 35 = 39.\n\nThey have \\boxed{39} chocolates left.",
            "predicted_answer": "39",
            "is_correct": True
        }
    ]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for item in test_data:
        temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    temp_file.close()
    
    return temp_file.name


def test_self_cot_functionality():
    """Test Self-CoT functionality."""
    print("Testing Self-CoT functionality...")
    
    # Create test Self-CoT data
    test_self_cot_path = create_test_self_cot_data()
    print(f"Created test Self-CoT data at: {test_self_cot_path}")
    
    try:
        # Create test configuration
        test_cfg = OmegaConf.create({
            "name": "gsm8k",
            "num_query_samples": 10,
            "seed": 42,
            "use_self_cot": True,
            "self_cot_path": test_self_cot_path,
            "training_mode": "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
        })
        
        # Test dataset initialization with Self-CoT
        print("Testing dataset initialization with Self-CoT...")
        dataset = GSM8KDataset(test_cfg, model_processor=None, model_name="qwen2.5-math-7b-instruct")
        
        print(f"Dataset state: {dataset.dataset_state}")
        print(f"Support set size: {len(dataset.support_set)}")
        print(f"Query set size: {len(dataset.query_set)}")
        
        # Test that Self-CoT data is loaded
        if hasattr(dataset, '_self_cot_data') and dataset._self_cot_data:
            print(f"Self-CoT data loaded: {len(dataset._self_cot_data)} samples")
        else:
            print("ERROR: Self-CoT data not loaded!")
            return False
        
        # Test that support set is filtered
        if hasattr(dataset, '_filtered_support_set') and dataset._filtered_support_set:
            print(f"Filtered support set: {len(dataset._filtered_support_set)} samples")
        else:
            print("ERROR: Support set not filtered!")
            return False
        
        # Test chat template creation for Self-CoT
        print("Testing chat template creation for Self-CoT...")
        sample = dataset.support_set[0]
        messages = dataset._create_qwen_chat_template(
            question=sample['question'],
            use_cot=True,
            few_shot_examples=[sample],
            dataset_state=DatasetState.TRAIN_TEACHER_SELF_COT
        )
        
        print(f"Generated {len(messages)} messages")
        for i, msg in enumerate(messages):
            print(f"Message {i}: {msg['role']} - {msg['content'][:100]}...")
        
        print("✅ All Self-CoT tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(test_self_cot_path):
            os.unlink(test_self_cot_path)
            print(f"Cleaned up temporary file: {test_self_cot_path}")


def test_dataset_state_enum():
    """Test the new DatasetState enum values."""
    print("Testing DatasetState enum...")
    
    # Test that new enum values exist
    expected_states = [
        "TRAIN_TEACHER_SELF_COT",
        "TRAIN_STUDENT_ONESHOT_SELF_COT", 
        "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
    ]
    
    for state_name in expected_states:
        if hasattr(DatasetState, state_name):
            state = getattr(DatasetState, state_name)
            print(f"✅ Found {state_name}: {state}")
        else:
            print(f"❌ Missing {state_name}")
            return False
    
    print("✅ All DatasetState enum tests passed!")
    return True


if __name__ == "__main__":
    print("Running Self-CoT functionality tests...\n")
    
    # Test 1: DatasetState enum
    test1_passed = test_dataset_state_enum()
    print()
    
    # Test 2: Self-CoT functionality
    test2_passed = test_self_cot_functionality()
    print()
    
    # Summary
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Self-CoT functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 