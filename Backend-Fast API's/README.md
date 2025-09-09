# FastAPI Object Detection Backend

A clean, working FastAPI backend for object detection using YOLOv8. This service accepts image uploads and returns detection results with bounding boxes and confidence scores.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python main.py
```

The server will start on `http://localhost:8000`

## 📡 API Endpoints

### Test Endpoint
- **GET** `/` - Returns API status confirmation

### Detection Endpoint  
- **POST** `/detect` - Upload image for object detection
  - **Input**: Image file via multipart/form-data
  - **Parameters**: 
    - `file`: Image file (required)
    - `confidence`: Confidence threshold 0.0-1.0 (optional, default: 0.7)

### Health Check
- **GET** `/health` - Returns system health status

## 📖 Usage Examples

### Test the API
```bash
curl http://localhost:8000/
```

### Upload Image for Detection
```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@your_image.jpg" \
     -F "confidence=0.7"
```

### Python Example
```python
import requests

# Test endpoint
response = requests.get("http://localhost:8000/")
print(response.json())

# Upload image for detection
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    data = {"confidence": 0.7}
    response = requests.post("http://localhost:8000/detect", files=files, data=data)
    print(response.json())
```

## 📋 Response Format

### Successful Detection Response
```json
{
  "detections": [
    {
      "class_name": "dog",
      "confidence": 0.95,
      "box": {
        "x1": 150, "y1": 200, "x2": 300, "y2": 450
      }
    }
  ],
  "total_detections": 1
}
```

## 🔧 Features

- ✅ **PyTorch 2.6+ Compatible** - Fixed compatibility issues
- ✅ **GPU/CPU Auto-detection** - Automatically uses available hardware
- ✅ **Image Upload Support** - Accepts JPG, PNG, JPEG, etc.
- ✅ **CORS Enabled** - Ready for mobile app integration
- ✅ **Error Handling** - Comprehensive validation and error responses
- ✅ **Health Monitoring** - Built-in health check endpoint

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🛠️ Troubleshooting

### Common Issues

1. **Model Download**: The YOLOv8 model will be automatically downloaded on first run
2. **Memory Issues**: The app uses YOLOv8n (nano) for faster inference and lower memory usage
3. **GPU Support**: Automatically detects and uses CUDA if available

### Logs
Check console output for:
- Model loading status
- Device detection (CPU/GPU)
- Inference results
- Error messages