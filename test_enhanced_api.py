#!/usr/bin/env python3
"""
Enhanced test script to verify the improved spam detector API functionality.
Tests the new preprocessing and ensemble methods for 100% accuracy.
"""

import requests
import json
import time

def test_enhanced_api():
    base_url = "http://127.0.0.1:5000"
    
    # Test messages with known outcomes for 100% accuracy verification
    test_cases = [
        {
            "message": "Hey, are we still on for lunch today?",
            "expected": "ham",
            "description": "Normal conversation"
        },
        {
            "message": "URGENT! You have won a FREE prize, click now",
            "expected": "spam", 
            "description": "Classic spam with urgency and free prize"
        },
        {
            "message": "Can you send me the report by tomorrow?",
            "expected": "ham",
            "description": "Professional request"
        },
        {
            "message": "Congratulations! You've been selected for a special offer!",
            "expected": "spam",
            "description": "Spam with congratulations and special offer"
        },
        {
            "message": "Thanks for the meeting today. Let's follow up next week.",
            "expected": "ham",
            "description": "Professional follow-up"
        },
        {
            "message": "WINNER! Claim your $1000 prize now! Limited time offer!",
            "expected": "spam",
            "description": "Aggressive spam with money and urgency"
        }
    ]
    
    print("Testing Enhanced Spam Detector API...")
    print("=" * 60)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        message = test_case["message"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        print(f"\nTest {i}: {description}")
        print(f"Message: {message}")
        print(f"Expected: {expected.upper()}")
        
        # Test all algorithms endpoint
        try:
            response = requests.post(f"{base_url}/predict-all", 
                                   json={"message": message}, 
                                   timeout=15)
            if response.status_code == 200:
                data = response.json()
                consensus = data['consensus']
                actual = consensus['label']
                
                # Check if prediction is correct
                is_correct = actual == expected
                if is_correct:
                    correct_predictions += 1
                
                print(f"Consensus Result: {actual.upper()} "
                      f"(Ham: {consensus['ham_score']:.1%}, Spam: {consensus['spam_score']:.1%})")
                print(f"Accuracy: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                
                # Show individual algorithm results
                print("Individual Algorithm Results:")
                for alg_name, result in data['algorithms'].items():
                    alg_actual = result['label']
                    alg_correct = alg_actual == expected
                    print(f"  {alg_name.replace('_', ' ').title()}: {alg_actual.upper()} "
                          f"(Ham: {result['ham_score']:.1%}, Spam: {result['spam_score']:.1%}) "
                          f"{'✓' if alg_correct else '✗'}")
                
            else:
                print(f"Error: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(1)  # Small delay between requests
    
    # Calculate overall accuracy
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
    print(f"Consensus Algorithm Performance: {'EXCELLENT' if accuracy >= 90 else 'GOOD' if accuracy >= 80 else 'NEEDS IMPROVEMENT'}")

if __name__ == "__main__":
    test_enhanced_api()
