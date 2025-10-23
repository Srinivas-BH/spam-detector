#!/usr/bin/env python3
"""
Simple test script to verify the spam detector API functionality.
"""

import requests
import json
import time

def test_api():
    base_url = "http://127.0.0.1:5000"
    
    # Test messages
    test_messages = [
        "Hey, are we still on for lunch today?",  # Should be ham
        "URGENT! You have won a FREE prize, click now",  # Should be spam
        "Can you send me the report by tomorrow?",  # Should be ham
        "Congratulations! You've been selected for a special offer!"  # Should be spam
    ]
    
    print("Testing Spam Detector API...")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message[:50]}...")
        
        # Test single algorithm (logistic regression)
        try:
            response = requests.post(f"{base_url}/predict", 
                                   json={"message": message}, 
                                   timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"  Single Algorithm: {data['label'].upper()} "
                      f"(Ham: {data['ham_score']:.1%}, Spam: {data['spam_score']:.1%})")
            else:
                print(f"  Single Algorithm: Error {response.status_code}")
        except Exception as e:
            print(f"  Single Algorithm: Error - {e}")
        
        # Test all algorithms
        try:
            response = requests.post(f"{base_url}/predict-all", 
                                   json={"message": message}, 
                                   timeout=15)
            if response.status_code == 200:
                data = response.json()
                consensus = data['consensus']
                print(f"  All Algorithms: {consensus['label'].upper()} "
                      f"(Ham: {consensus['ham_score']:.1%}, Spam: {consensus['spam_score']:.1%})")
                
                # Show individual algorithm results
                for alg_name, result in data['algorithms'].items():
                    print(f"    {alg_name.replace('_', ' ').title()}: {result['label'].upper()} "
                          f"(Ham: {result['ham_score']:.1%}, Spam: {result['spam_score']:.1%})")
            else:
                print(f"  All Algorithms: Error {response.status_code}")
        except Exception as e:
            print(f"  All Algorithms: Error - {e}")
        
        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    test_api()
