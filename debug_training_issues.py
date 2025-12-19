"""
Debug script để kiểm tra các vấn đề training và rerank.

Chạy script này để kiểm tra:
1. Training loss có giảm không
2. Letter tokens có được tìm thấy không
3. Model prediction có uniform không
4. Evaluation setup có đúng không
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from rerank.models.llm import LLMModel, LETTERS, build_prompt_from_candidates

def test_letter_token_extraction():
    """Test xem letter tokens có được tìm thấy không."""
    print("=" * 80)
    print("TEST 1: Letter Token Extraction")
    print("=" * 80)
    
    model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    print(f"Loading tokenizer: {model_name}")
    
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying standard tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    print(f"\nTokenizer vocab size: {len(tokenizer)}")
    print(f"UNK token ID: {tokenizer.unk_token_id}")
    
    # Test letter tokens
    print("\nTesting letter tokens:")
    found_tokens = []
    missing_tokens = []
    
    for i, letter in enumerate(LETTERS[:26]):  # Test A-Z
        # Strategy 1: Direct
        token_id = tokenizer.convert_tokens_to_ids(letter)
        if token_id != tokenizer.unk_token_id:
            found_tokens.append((letter, token_id, "direct"))
            continue
        
        # Strategy 2: Space prefix
        token_id = tokenizer.convert_tokens_to_ids(" " + letter)
        if token_id != tokenizer.unk_token_id:
            found_tokens.append((letter, token_id, "space_prefix"))
            continue
        
        # Strategy 3: Encoding
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) > 0 and encoded[0] != tokenizer.unk_token_id:
            found_tokens.append((letter, encoded[0], "encoding"))
            continue
        
        missing_tokens.append(letter)
    
    print(f"\n✅ Found {len(found_tokens)}/26 letter tokens (A-Z)")
    if found_tokens:
        print(f"Sample found tokens: {found_tokens[:5]}")
    
    if missing_tokens:
        print(f"❌ Missing tokens: {missing_tokens[:10]}")
        return False
    else:
        print("✅ All letter tokens found!")
        return True


def test_model_prediction():
    """Test model prediction trên một sample."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Prediction")
    print("=" * 80)
    
    # Create a simple test prompt
    prompt = """You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2

Candidate items:
A. candidate1
B. candidate2
C. candidate3
D. candidate4
E. candidate5

Answer with only one letter (A-E)."""
    
    print("Test prompt:")
    print(prompt)
    print()
    
    # Load model
    print("Loading model...")
    try:
        model = LLMModel(train_data=None, model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
        model.load_model(use_torch_compile=False)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Predict
    print("\nPredicting probabilities...")
    try:
        probs = model.predict_probs(prompt, num_candidates=5)
        print(f"Probabilities: {probs}")
        print(f"Max prob: {np.max(probs):.4f}, Min prob: {np.min(probs):.4f}")
        print(f"Std: {np.std(probs):.4f}")
        print(f"Predicted letter: {LETTERS[np.argmax(probs)]}")
        
        # Check if uniform
        expected_uniform = 1.0 / len(probs)
        if np.std(probs) < 0.01:
            print(f"\n⚠️  WARNING: Probabilities are nearly uniform!")
            print(f"   Expected uniform: {expected_uniform:.4f}")
            print(f"   Actual std: {np.std(probs):.4f}")
            return False
        else:
            print(f"\n✅ Probabilities are not uniform (std={np.std(probs):.4f})")
            return True
    except Exception as e:
        print(f"❌ Error predicting: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_data_format():
    """Test training data format."""
    print("\n" + "=" * 80)
    print("TEST 3: Training Data Format")
    print("=" * 80)
    
    # Sample training data
    sample_data = {
        "messages": [
            {
                "role": "user",
                "content": """You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2

Candidate items:
A. candidate1
B. candidate2
C. candidate3

Answer with only one letter (A-C)."""
            },
            {
                "role": "assistant",
                "content": "B"  # Letter label
            }
        ]
    }
    
    print("Sample training data:")
    print(f"User message length: {len(sample_data['messages'][0]['content'])}")
    print(f"Assistant message: {sample_data['messages'][1]['content']}")
    print(f"Target is letter: {sample_data['messages'][1]['content'] in LETTERS}")
    
    # Check tokenization
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            sample_data["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        print(f"\nTokenized length: {tokens['input_ids'].shape[1]}")
        print(f"Last 10 tokens: {tokens['input_ids'][0][-10:].tolist()}")
        
        # Check if target letter is in tokens
        target_letter = sample_data["messages"][1]["content"]
        target_encoded = tokenizer.encode(target_letter, add_special_tokens=False)
        print(f"Target letter '{target_letter}' encoded as: {target_encoded}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_setup():
    """Test evaluation setup."""
    print("\n" + "=" * 80)
    print("TEST 4: Evaluation Setup")
    print("=" * 80)
    
    # Simulate evaluation
    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    gt_items = [5]  # Ground truth
    
    # Check if GT is in candidates
    if gt_items[0] in candidates:
        print(f"✅ GT item {gt_items[0]} is in candidates")
    else:
        print(f"❌ GT item {gt_items[0]} is NOT in candidates!")
        return False
    
    # Simulate uniform prediction (random)
    probs = np.ones(len(candidates)) / len(candidates)
    ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
    top_20 = [item_id for item_id, _ in ranked[:20]]
    
    hits = len(set(top_20) & set(gt_items))
    recall = hits / len(gt_items)
    
    print(f"Random baseline recall@20: {recall:.4f}")
    print(f"Expected random: {20 / len(candidates):.4f}")
    
    if abs(recall - (20 / len(candidates))) < 0.1:
        print("✅ Evaluation setup looks correct")
        return True
    else:
        print("⚠️  Evaluation setup may have issues")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("DEBUGGING TRAINING AND RERANK ISSUES")
    print("=" * 80)
    print()
    
    results = {}
    
    # Test 1: Letter token extraction
    results["letter_tokens"] = test_letter_token_extraction()
    
    # Test 2: Model prediction
    results["model_prediction"] = test_model_prediction()
    
    # Test 3: Training data format
    results["training_data"] = test_training_data_format()
    
    # Test 4: Evaluation setup
    results["evaluation"] = test_evaluation_setup()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed! Issues may be elsewhere.")
    else:
        print("\n❌ Some tests failed. Check the issues above.")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if not results["letter_tokens"]:
        print("1. ❌ Letter tokens not found - This is CRITICAL!")
        print("   → Model cannot predict letters → fallback to uniform → recall = random")
        print("   → Check tokenizer compatibility")
    
    if not results["model_prediction"]:
        print("2. ❌ Model predicts uniform distribution")
        print("   → Model chưa học được gì → recall = random")
        print("   → Check training loss có giảm không")
        print("   → Check training data format")
    
    if not results["training_data"]:
        print("3. ❌ Training data format có vấn đề")
        print("   → Check target labels có đúng format không")
    
    if not results["evaluation"]:
        print("4. ❌ Evaluation setup có vấn đề")
        print("   → Check GT items có trong candidates không")


if __name__ == "__main__":
    main()

