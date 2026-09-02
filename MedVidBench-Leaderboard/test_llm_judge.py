#!/usr/bin/env python3
"""
Test script to verify LLM judge is working correctly.
Run this after setting OPENAI_API_KEY.
"""
import os
import sys
import json

# Add evaluation directory to path
sys.path.insert(0, 'evaluation')

def test_api_key():
    """Check if API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        print("\nSet it with:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return False

    print(f"✓ API key found (length: {len(api_key)})")
    return True

def test_llm_judge():
    """Test LLM judge with a single sample"""
    try:
        from eval_caption_llm_judge import evaluate_single_caption_llm

        # Create a test sample
        test_data = {
            "prediction": "The surgeon performs a laparoscopic cholecystectomy using graspers and scissors to dissect the gallbladder from the liver bed.",
            "answer": "The surgeon removes the gallbladder using minimally invasive techniques with specialized instruments.",
            "metadata": {"video_id": "test_video"}
        }

        print("\nTesting LLM judge with sample caption...")
        print(f"Prediction: {test_data['prediction'][:100]}...")

        api_key = os.getenv('OPENAI_API_KEY')
        result = evaluate_single_caption_llm(
            test_data,
            task_type="video_summary",
            api_key=api_key
        )

        if result:
            print("\n✅ LLM judge working!")
            print(f"  Average Score: {result.get('average_score', 0):.3f}/5.0")
            print(f"  Aspect Scores:")
            for aspect, score in sorted(result.get('aspect_scores', {}).items()):
                print(f"    {aspect}: {score:.1f}/5.0")
            return True
        else:
            print("❌ LLM judge returned no result")
            return False

    except Exception as e:
        print(f"❌ Error testing LLM judge: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("MedVidBench LLM Judge Test")
    print("="*60)

    # Step 1: Check API key
    if not test_api_key():
        sys.exit(1)

    # Step 2: Test LLM judge
    if not test_llm_judge():
        print("\n⚠️  LLM judge test failed")
        print("This could mean:")
        print("  1. Invalid API key")
        print("  2. No OpenAI credits")
        print("  3. Network connectivity issue")
        print("  4. Import error")
        sys.exit(1)

    print("\n" + "="*60)
    print("✅ All tests passed! LLM judge is ready.")
    print("="*60)

if __name__ == "__main__":
    main()
