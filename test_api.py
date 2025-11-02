import requests
import json
import sys

def test_api():
    base_url = "http://127.0.0.1:5000"
    hello_url = f"{base_url}/hello"
    predict_url = f"{base_url}/api/predict"
    
    print("--- Starting Full API Test ---")
    
    # --- Step 1: Check the /hello debug route ---
    print(f"\n[Test 1] Checking server status at: {hello_url}")
    try:
        response = requests.get(hello_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('message') == 'Hello, the new server is working!':
            print("  ✓ SUCCESS: Server is running the new app.py code.")
        else:
            print("  ✗ FAILED: Server responded, but with unexpected data.")
            sys.exit()
            
    except requests.exceptions.JSONDecodeError:
        print("  ✗ FAILED: Server is returning HTML, not JSON.")
        print("  This means the '/hello' route is missing. The server is running OLD code.")
        print(f"  Response Snippet: {response.text[:100]}...")
        sys.exit()
    except requests.exceptions.HTTPError:
        print("  ✗ FAILED: Server returned a 404 or 500 error.")
        print("  This means the '/hello' route is missing. The server is running OLD code.")
        sys.exit()
    except requests.exceptions.ConnectionError:
        print("  ✗ FAILED: Could not connect to the server.")
        print("  Please make sure 'python app.py' is running in another terminal.")
        sys.exit()
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
        sys.exit()

    # --- Step 2: Test the /api/predict route ---
    print(f"\n[Test 2] Testing spam prediction at: {predict_url}")
    test_message = "Your account has been temporarily locked due to unusual activity. Please verify your identity immediately: http://bit.ly/secure-auth-123"
    
    try:
        payload = {'message': test_message}
        response = requests.post(predict_url, json=payload, timeout=10)
        response.raise_for_status() 
        data = response.json() 
        
        print("  ✓ SUCCESS: API Test Successful.")
        print(f"  Prediction: {data['prediction'].upper()} "
              f"(Spam Probability: {data['spam_probability']:.1%})")
              
    except requests.exceptions.JSONDecodeError:
        print("  ✗ FAILED: JSONDecodeError. This is a critical error.")
        print("  The server is returning HTML for '/api/predict'.")
        print("  This means there is an internal error in the 'predict_api' function.")
        print(f"  Response Snippet: {response.text[:100]}...")
        
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_api()