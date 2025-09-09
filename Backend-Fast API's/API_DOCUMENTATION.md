# FastAPI Object Detection API Documentation

## Base URL
```
http://localhost:8000
```

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "cuda_available": false,
  "model_info": {
    "total_classes": 80,
    "model_type": "YOLOv8s"
  }
}
```

### 2. Get Available Classes
**GET** `/classes`

Get all object classes that can be detected.

**Response:**
```json
{
  "success": true,
  "message": "Available 80 object classes",
  "total_classes": 80,
  "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
  "categories": {
    "animals": ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "objects": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
  }
}
```

### 3. Object Detection (Standard)
**POST** `/detect`

Detect objects in uploaded image with standard confidence threshold.

**Parameters:**
- `file` (multipart/form-data): Image file (jpg, png, jpeg, etc.)
- `confidence` (query parameter): Confidence threshold (0.0-1.0, default: 0.7)

**Response:**
```json
{
  "success": true,
  "message": "Successfully detected 2 object(s)",
  "detections": [
    {
      "class_name": "dog",
      "confidence": 0.95,
      "box": {
        "x1": 150,
        "y1": 200,
        "x2": 300,
        "y2": 450
      }
    },
    {
      "class_name": "person",
      "confidence": 0.88,
      "box": {
        "x1": 50,
        "y1": 100,
        "x2": 250,
        "y2": 400
      }
    }
  ],
  "total_detections": 2,
  "image_info": {
    "filename": "test_image.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 245760,
    "dimensions": {
      "width": 640,
      "height": 480
    },
    "format": "JPEG",
    "mode": "RGB"
  },
  "processing_time": 0.234
}
```

### 4. Object Detection (Lower Confidence)
**POST** `/detect-all`

Detect objects with lower confidence threshold for better detection.

**Parameters:**
- `file` (multipart/form-data): Image file (jpg, png, jpeg, etc.)
- `confidence` (query parameter): Confidence threshold (0.0-1.0, default: 0.3)

**Response:** Same format as `/detect`

## Error Responses

### No Objects Detected
```json
{
  "success": true,
  "message": "No objects detected. Try lowering the confidence threshold or use a different image.",
  "detections": [],
  "total_detections": 0,
  "image_info": {
    "filename": "test_image.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 245760,
    "dimensions": {
      "width": 640,
      "height": 480
    },
    "format": "JPEG",
    "mode": "RGB"
  },
  "processing_time": 0.156
}
```

### Validation Error
```json
{
  "success": false,
  "message": "Confidence must be between 0.0 and 1.0",
  "detections": [],
  "total_detections": 0,
  "image_info": {},
  "processing_time": 0.001
}
```

### File Error
```json
{
  "success": false,
  "message": "Error: Invalid file type: text/plain. Only image files are allowed.",
  "detections": [],
  "total_detections": 0,
  "image_info": {},
  "processing_time": 0.002
}
```

## React Native Usage Examples

### 1. Basic Detection
```javascript
const detectObjects = async (imageUri, confidence = 0.7) => {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'image.jpg',
  });

  try {
    const response = await fetch(`http://localhost:8000/detect?confidence=${confidence}`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    const result = await response.json();
    
    if (result.success) {
      console.log('Detections:', result.detections);
      console.log('Total:', result.total_detections);
      console.log('Processing time:', result.processing_time);
    } else {
      console.error('Error:', result.message);
    }
    
    return result;
  } catch (error) {
    console.error('Network error:', error);
    return {
      success: false,
      message: 'Network error',
      detections: [],
      total_detections: 0,
      image_info: {},
      processing_time: 0
    };
  }
};
```

### 2. Better Detection (Lower Confidence)
```javascript
const detectObjectsBetter = async (imageUri) => {
  return await detectObjects(imageUri, 0.3); // Use detect-all endpoint
};
```

### 3. Health Check
```javascript
const checkAPIHealth = async () => {
  try {
    const response = await fetch('http://localhost:8000/health');
    const health = await response.json();
    
    if (health.status === 'healthy' && health.model_loaded) {
      console.log('API is ready');
      return true;
    } else {
      console.log('API not ready:', health);
      return false;
    }
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};
```

### 4. Get Available Classes
```javascript
const getAvailableClasses = async () => {
  try {
    const response = await fetch('http://localhost:8000/classes');
    const classes = await response.json();
    
    if (classes.success) {
      console.log('Available classes:', classes.classes);
      console.log('Animals:', classes.categories.animals);
      console.log('Vehicles:', classes.categories.vehicles);
      return classes;
    }
  } catch (error) {
    console.error('Failed to get classes:', error);
    return null;
  }
};
```

## Response Validation in React Native

```javascript
const validateDetectionResponse = (response) => {
  // Check if response is valid
  if (!response || typeof response !== 'object') {
    return { valid: false, error: 'Invalid response format' };
  }

  // Check required fields
  const requiredFields = ['success', 'message', 'detections', 'total_detections', 'image_info'];
  for (const field of requiredFields) {
    if (!(field in response)) {
      return { valid: false, error: `Missing field: ${field}` };
    }
  }

  // Check success status
  if (!response.success) {
    return { valid: false, error: response.message };
  }

  // Check detections format
  if (!Array.isArray(response.detections)) {
    return { valid: false, error: 'Detections must be an array' };
  }

  // Validate each detection
  for (const detection of response.detections) {
    if (!detection.class_name || !detection.confidence || !detection.box) {
      return { valid: false, error: 'Invalid detection format' };
    }
    
    if (detection.confidence < 0 || detection.confidence > 1) {
      return { valid: false, error: 'Invalid confidence score' };
    }
  }

  return { valid: true, data: response };
};
```

## Tips for Better Detection

1. **Use appropriate confidence thresholds:**
   - `0.7-0.9`: High confidence, fewer false positives
   - `0.3-0.5`: Balanced detection
   - `0.1-0.3`: Maximum detection, may include false positives

2. **Image requirements:**
   - Supported formats: JPG, PNG, JPEG, BMP, TIFF, WEBP
   - Good lighting and contrast
   - Objects should be clearly visible
   - Avoid very small or very large images

3. **Common detectable objects:**
   - People, animals (dog, cat, bird, etc.)
   - Vehicles (car, bus, truck, etc.)
   - Common objects (bottle, chair, laptop, etc.)

4. **Error handling:**
   - Always check `response.success` before processing
   - Handle network errors gracefully
   - Provide user feedback for different error types
















API: Return Annotated Image (boxes + labels)
URL: POST http://<SERVER_HOST>:8000/detect-image
Purpose: Detect objects and return an annotated image (PNG) with bounding boxes and labels drawn
Auth: None
Consumes: multipart/form-data
Produces: image/png (on success), application/json (on error)
Parameters:
file (form-data, required): Image file. Accepted types: image/jpeg, image/png, image/bmp, image/tiff, image/webp.
confidence (query, optional): float in [0.0, 1.0]. Default: 0.3. Lower values produce more boxes.
Successful Response:
Status: 200 OK
Body: image/png
Headers:
X-Detections-Count: number of boxes drawn (string)
X-Processing-Time: seconds to process (string e.g. "0.245")
X-Image-Width: original width (string)
X-Image-Height: original height (string)
Zero detections:
Still returns image/png (original image), with X-Detections-Count: "0".
Error Responses:
400 Bad Request (application/json)
Invalid file type
Missing file
Invalid confidence (outside 0.0–1.0)
Example JSON:
{"detail":"Invalid file type: None. Only image files are allowed."}
500 Internal Server Error (application/json)
Unexpected server/inference error

Example cURL:
curl -X POST "http://localhost:8000/detect-image?confidence=0.3" \
  -H "accept: image/png" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg" \
  --output annotated.png -i

Goal: Display returned PNG; read detection metadata from headers for UX logic
