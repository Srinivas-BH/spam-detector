import requests
import json
import time

def test_enhanced_api():
    # Use the CORRECT API endpoint
    base_url = "http://127.0.0.1:5000/api/predict"
    
    test_cases = [
        {"message": "Hey, are we still on for lunch today?", "expected": "ham"},
        {"message": "URGENT! You have won a FREE prize, click now", "expected": "spam"},
        {"message": "Can you send me the report by tomorrow?", "expected": "ham"},
        {"message": "Congratulations! You've been selected for a special offer!", "expected": "spam"},
        {"message": "WINNER! Claim your $1000 prize now! Limited time offer!", "expected": "spam"},
        {"message": "Your Google verification code is 845210.", "expected": "ham"}
    ]
    
    print("Testing Enhanced Spam Detector API...")
    print("=" * 60)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        message = test_case["message"]
        expected = test_case["expected"]
        
        print(f"\nTest {i}: {message[:40]}...")
        print(f"Expected: {expected.upper()}")
        
        try:
            payload = {'message': message}
            response = requests.post(base_url, json=payload, timeout=10)
            
            response.raise_for_status() # Raise error for bad responses
            data = response.json() # This is the line that fails
            actual = data['prediction']
            
            is_correct = actual == expected
            if is_correct:
                correct_predictions += 1
            
            print(f"  Actual: {actual.upper()} (Prob: {data['spam_probability']:.1%})")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                
        except requests.exceptions.JSONDecodeError:
            print("  --- TEST FAILED ---")
            print("  Could not decode JSON. The server sent back HTML.")
            print(f"  Response Text: {response.text[:100]}...") # Show HTML snippet
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(0.5)
    
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.1f}% ({correct_predictions}/{total_tests})")

if __name__ == "__main__":
    test_enhanced_api()