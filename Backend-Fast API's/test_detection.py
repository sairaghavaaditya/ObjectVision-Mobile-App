"""
Test script to help debug detection issues
"""

import requests
import json

def test_detection_with_different_confidence():
    """Test detection with different confidence levels"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Object Detection with Different Confidence Levels")
    print("=" * 60)
    
    # Test image file
    test_image = "test_image.jpg"
    
    confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for confidence in confidence_levels:
        print(f"\n🔍 Testing with confidence threshold: {confidence}")
        print("-" * 40)
        
        try:
            with open(test_image, "rb") as f:
                files = {"file": f}
                data = {"confidence": confidence}
                response = requests.post(f"{base_url}/detect", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Status: {response.status_code}")
                    print(f"📊 Total detections: {result['total_detections']}")
                    
                    if result['detections']:
                        for i, detection in enumerate(result['detections']):
                            print(f"   Detection {i+1}: {detection['class_name']} "
                                  f"(confidence: {detection['confidence']:.3f})")
                    else:
                        print("   No detections found")
                else:
                    print(f"❌ Error: {response.status_code}")
                    print(f"   Response: {response.json()}")
                    
        except FileNotFoundError:
            print(f"❌ Test image '{test_image}' not found")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("💡 Tips for better detection:")
    print("1. Try lower confidence thresholds (0.1-0.3)")
    print("2. Use the /detect-all endpoint with confidence=0.3")
    print("3. Check server logs for raw detection details")
    print("4. Ensure images contain common objects (person, car, dog, etc.)")

def test_detect_all_endpoint():
    """Test the new detect-all endpoint"""
    base_url = "http://localhost:8000"
    
    print("\n🔍 Testing /detect-all endpoint (lower confidence)")
    print("-" * 50)
    
    try:
        with open("test_image.jpg", "rb") as f:
            files = {"file": f}
            data = {"confidence": 0.3}
            response = requests.post(f"{base_url}/detect-all", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"📊 Total detections: {result['total_detections']}")
                
                if result['detections']:
                    for i, detection in enumerate(result['detections']):
                        print(f"   Detection {i+1}: {detection['class_name']} "
                              f"(confidence: {detection['confidence']:.3f})")
                else:
                    print("   No detections found")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.json()}")
                
    except FileNotFoundError:
        print("❌ Test image 'test_image.jpg' not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_detection_with_different_confidence()
    test_detect_all_endpoint()

