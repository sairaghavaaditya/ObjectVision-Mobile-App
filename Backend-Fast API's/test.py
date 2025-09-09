"""
Simple test script for the FastAPI Object Detection Backend
"""

import requests
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing FastAPI Object Detection Backend")
    print("=" * 50)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed. Make sure the server is running.")
        return
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Detection endpoint (if test image exists)
    print("\n3. Testing detection endpoint...")
    try:
        with open("test_image.jpg", "rb") as f:
            files = {"file": f}
            data = {"confidence": 0.7}
            response = requests.post(f"{base_url}/detect", files=files, data=data)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Total detections: {result['total_detections']}")
                for i, detection in enumerate(result['detections']):
                    print(f"   Detection {i+1}: {detection['class_name']} "
                          f"(confidence: {detection['confidence']:.2f})")
            else:
                print(f"   Error response: {response.json()}")
    except FileNotFoundError:
        print("   ⚠️  No test_image.jpg found. Create an image file to test detection.")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing complete!")

if __name__ == "__main__":
    test_api()

