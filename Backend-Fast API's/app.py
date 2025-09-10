"""
FastAPI Object Detection Backend - Final Working Version
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
from torchvision import transforms, models
import torch.nn.functional as F
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model: Optional[YOLO] = None
classifier_model: Optional[torch.nn.Module] = None
imagenet_idx_to_label: Optional[dict] = None
device: Optional[torch.device] = None


# Fix PyTorch 2.6+ compatibility issue
def fix_pytorch_compatibility():
    """Fix PyTorch 2.6+ weights_only compatibility for YOLOv8"""
    try:
        # Monkey patch torch.load to use weights_only=False
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        logger.info("PyTorch compatibility fix applied successfully")
        
    except Exception as e:
        logger.warning(f"Could not apply PyTorch compatibility fix: {e}")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup, cleanup on shutdown"""
    global model, device
    global classifier_model, imagenet_idx_to_label
    
    # Startup
    logger.info("Starting up FastAPI Object Detection Backend...")
    
    # Apply PyTorch compatibility fix
    fix_pytorch_compatibility()
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load YOLOv8 model - try different models for better accuracy
    try:
        # Try yolov8s (small) first for better accuracy, fallback to nano
        try:
            model = YOLO('yolov8s.pt')
            logger.info("Loaded YOLOv8s (small) model for better accuracy")
        except:
            model = YOLO('yolov8n.pt')
            logger.info("Loaded YOLOv8n (nano) model")
        
        model.to(device)
        logger.info("YOLOv8 model loaded successfully")
        logger.info(f"Model can detect {len(model.model.names)} classes")
        logger.info(f"Available classes: {list(model.model.names.values())[:10]}...")

        # Load Imagenet classifier (ResNet50) lazily at startup
        try:
            classifier_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            classifier_model.eval()
            classifier_model.to(device)
            # Build index to label map from weights metadata
            weights = models.ResNet50_Weights.DEFAULT
            imagenet_idx_to_label = {i: c for i, c in enumerate(weights.meta["categories"])}
            logger.info("ResNet50 ImageNet classifier loaded")
        except Exception as e:
            logger.warning(f"Failed to load ImageNet classifier: {e}")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI Object Detection Backend...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Object Detection API",
    description="FastAPI backend for YOLOv8 object detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Detections-Count",
        "X-Processing-Time",
        "X-Image-Width",
        "X-Image-Height",
        "X-Cls-Label",
        "X-Cls-Prob",
    ],
)


# Pydantic models
class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    class_name: str
    confidence: float
    box: DetectionBox


class DetectionResponse(BaseModel):
    success: bool
    message: str
    detections: List[Detection]
    total_detections: int
    image_info: dict
    processing_time: Optional[float] = None


# Utility functions
def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only image files are allowed."
        )
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_extension = file.filename.lower().split('.')[-1] if file.filename else ''
    if f'.{file_extension}' not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )


def process_uploaded_image(file: UploadFile) -> tuple[np.ndarray, dict]:
    """Process uploaded image file and return as numpy array with image info"""
    try:
        # Read image data
        image_data = file.file.read()
        
        # Open with PIL
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Create image info
        image_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(image_data),
            "dimensions": {
                "width": image.size[0],
                "height": image.size[1]
            },
            "format": image.format,
            "mode": image.mode
        }
        
        logger.info(f"Processed image: {image.size} -> {image_array.shape}")
        return image_array, image_info
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )


def run_inference(image_array: np.ndarray, confidence_threshold: float = 0.7) -> List[Detection]:
    """Run YOLOv8 inference on image array"""
    global model, device
    
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please restart the server."
        )
    
    try:
        # Run inference with lower default confidence for better detection
        results = model.predict(
            source=image_array,
            conf=confidence_threshold,
            device=device,
            verbose=False,
            stream=False,
            save=False,
            show=False
        )
        
        detections = []
        all_detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Log all detections before filtering
            if result.boxes is not None:
                logger.info(f"Raw detections found: {len(result.boxes)}")
                
                for i, box in enumerate(result.boxes):
                    # Extract coordinates
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                    confidence = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    class_name = model.model.names[class_id]
                    
                    # Log all detections
                    all_detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2]
                    })
                    
                    logger.info(f"Detection {i+1}: {class_name} (confidence: {confidence:.3f})")
                    
                    # Only add if confidence meets threshold
                    if confidence >= confidence_threshold:
                        detection = Detection(
                            class_name=class_name,
                            confidence=confidence,
                            box=DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        )
                        detections.append(detection)
            else:
                logger.info("No detections found in image")
        
        logger.info(f"Inference complete: {len(detections)} detections above threshold {confidence_threshold}")
        logger.info(f"Total raw detections: {len(all_detections)}")
        
        return detections
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


def draw_annotated_image(image_array: np.ndarray, detections: List[Detection]) -> io.BytesIO:
    """Draw bounding boxes and labels on the image and return PNG bytes buffer."""
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)

    # Try to load a truetype font; fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det.box.x1, det.box.y1, det.box.x2, det.box.y2
        label = f"{det.class_name}: {det.confidence:.2f}"

        # Box
        draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=3)

        # Label background
        text_size = draw.textbbox((x1, y1), label, font=font)
        tw = text_size[2] - text_size[0]
        th = text_size[3] - text_size[1]
        tx, ty = x1, max(0, y1 - th - 4)
        draw.rectangle([(tx, ty), (tx + tw + 6, ty + th + 2)], fill="lime")
        # Label text
        draw.text((tx + 3, ty + 1), label, fill="black", font=font)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf

def resize_for_detection(image_array: np.ndarray, target_long_side: int) -> tuple[np.ndarray, dict]:
    """Resize keeping aspect ratio so the longer side equals target_long_side. Returns resized array and info."""
    if target_long_side <= 0:
        h, w = image_array.shape[:2]
        return image_array, {"width": w, "height": h, "scale": 1.0}
    h, w = image_array.shape[:2]
    long_side = max(h, w)
    if long_side == target_long_side:
        return image_array, {"width": w, "height": h, "scale": 1.0}
    scale = target_long_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pil_img = Image.fromarray(image_array).resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil_img), {"width": new_w, "height": new_h, "scale": scale}


def run_classification(image_array: np.ndarray, topk: int = 5, mode: str = "warp") -> dict:
    """Run ImageNet classification and return top-1 and top-k results.
    mode: "warp" resizes to 224x224 without cropping; "crop" uses 256->center crop 224.
    """
    global classifier_model, device, imagenet_idx_to_label
    if classifier_model is None:
        raise HTTPException(status_code=500, detail="Classifier not loaded")

    if mode == "crop":
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    try:
        with torch.no_grad():
            input_tensor = preprocess(image_array).unsqueeze(0).to(device)
            logits = classifier_model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            top_indices = np.argsort(probs)[-topk:][::-1].tolist()
            top = []
            for idx in top_indices:
                label = imagenet_idx_to_label.get(idx, str(idx)) if imagenet_idx_to_label else str(idx)
                top.append({"index": int(idx), "label": label, "probability": float(probs[idx])})
            top1 = top[0] if len(top) > 0 else {"index": -1, "label": "unknown", "probability": 0.0}
            return {"top1": top1, "topk": top, "mode": mode}
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# API Endpoints
@app.get("/")
async def test_endpoint():
    """Test endpoint to verify API is running"""
    return {"message": "API is running!"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = 0.7,
    detect_size: int = 0
):
    """
    Detect objects in uploaded image using YOLOv8
    
    - **file**: Image file (jpg, png, jpeg, etc.)
    - **confidence**: Minimum confidence threshold (0.0-1.0)
    
    Returns list of detected objects with bounding boxes and confidence scores.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate confidence parameter
        if not 0.0 <= confidence <= 1.0:
            return DetectionResponse(
                success=False,
                message="Confidence must be between 0.0 and 1.0",
                detections=[],
                total_detections=0,
                image_info={},
                processing_time=time.time() - start_time
            )
        
        # Validate file
        validate_image_file(file)
        
        # Process image
        image_array, image_info = process_uploaded_image(file)
        # Optional resize for detection
        resized_array, resize_info = resize_for_detection(image_array, detect_size)
        if detect_size > 0:
            image_array = resized_array
            image_info["resized_dimensions"] = {"width": resize_info["width"], "height": resize_info["height"], "scale": resize_info["scale"]}
        
        # Run inference
        detections = run_inference(image_array, confidence)
        
        # Determine success message
        if len(detections) > 0:
            message = f"Successfully detected {len(detections)} object(s)"
        else:
            message = "No objects detected. Try lowering the confidence threshold or use a different image."
        
        # Return response
        return DetectionResponse(
            success=True,
            message=message,
            detections=detections,
            total_detections=len(detections),
            image_info=image_info,
            processing_time=time.time() - start_time
        )
        
    except HTTPException as e:
        return DetectionResponse(
            success=False,
            message=f"Error: {e.detail}",
            detections=[],
            total_detections=0,
            image_info={},
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Unexpected error in detect_objects: {e}")
        return DetectionResponse(
            success=False,
            message=f"Unexpected error: {str(e)}",
            detections=[],
            total_detections=0,
            image_info={},
            processing_time=time.time() - start_time
        )


# @app.post("/detect-all", response_model=DetectionResponse)
# async def detect_objects_all(
#     file: UploadFile = File(..., description="Image file to analyze"),
#     confidence: float = 0.3
# ):
#     """
#     Detect objects in uploaded image using YOLOv8 with lower confidence threshold
    
#     - **file**: Image file (jpg, png, jpeg, etc.)
#     - **confidence**: Minimum confidence threshold (0.0-1.0, default: 0.3)
    
#     Returns list of detected objects with bounding boxes and confidence scores.
#     This endpoint uses a lower default confidence for better detection.
#     """
#     import time
#     start_time = time.time()
    
#     try:
#         # Validate confidence parameter
#         if not 0.0 <= confidence <= 1.0:
#             return DetectionResponse(
#                 success=False,
#                 message="Confidence must be between 0.0 and 1.0",
#                 detections=[],
#                 total_detections=0,
#                 image_info={},
#                 processing_time=time.time() - start_time
#             )
        
#         # Validate file
#         validate_image_file(file)
        
#         # Process image
#         image_array, image_info = process_uploaded_image(file)
        
#         # Run inference with lower confidence
#         detections = run_inference(image_array, confidence)
        
#         # Determine success message
#         if len(detections) > 0:
#             message = f"Successfully detected {len(detections)} object(s) with lower confidence threshold"
#         else:
#             message = "No objects detected even with lower confidence. The image might not contain detectable objects."
        
#         # Return response
#         return DetectionResponse(
#             success=True,
#             message=message,
#             detections=detections,
#             total_detections=len(detections),
#             image_info=image_info,
#             processing_time=time.time() - start_time
#         )

#     except HTTPException as e:
#         return DetectionResponse(
#             success=False,
#             message=f"Error: {e.detail}",
#             detections=[],
#             total_detections=0,
#             image_info={},
#             processing_time=time.time() - start_time
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error in detect_objects_all: {e}")
#         return DetectionResponse(
#             success=False,
#             message=f"Unexpected error: {str(e)}",
#             detections=[],
#             total_detections=0,
#             image_info={},
#             processing_time=time.time() - start_time
#         )

@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = 0.3,
    classify_mode: str = "warp",
    detect_size: int = 0
):
    """
    Detect objects and return an annotated image (PNG) with boxes and labels drawn.
    Returns image/png with detection metadata in headers for convenience.
    """
    import time
    start_time = time.time()

    # Validate confidence parameter
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")

    # Validate and process image
    validate_image_file(file)
    image_array, image_info = process_uploaded_image(file)

    # Optional resize for detection
    resized_array, resize_info = resize_for_detection(image_array, detect_size)
    detect_array = resized_array if detect_size > 0 else image_array

    # Inference
    detections = run_inference(detect_array, confidence)

    # Classification (best-effort)
    cls_label = None
    cls_prob = None
    cls_mode = classify_mode
    try:
        cls = run_classification(image_array, topk=5, mode=classify_mode if classify_mode in ("warp", "crop") else "warp")
        cls_label = cls.get("top1", {}).get("label")
        cls_prob = cls.get("top1", {}).get("probability")
        cls_mode = cls.get("mode", cls_mode)
    except Exception as e:
        logger.warning(f"Classification skipped: {e}")

    # Draw
    buf = draw_annotated_image(detect_array, detections)

    # Build response with useful headers for the app
    headers = {
        "X-Detections-Count": str(len(detections)),
        "X-Processing-Time": f"{time.time() - start_time:.3f}",
        "X-Image-Width": str(image_info.get("dimensions", {}).get("width", 0)),
        "X-Image-Height": str(image_info.get("dimensions", {}).get("height", 0)),
    }
    if cls_label is not None:
        headers["X-Cls-Label"] = cls_label
    if cls_prob is not None:
        headers["X-Cls-Prob"] = f"{cls_prob:.4f}"
    if cls_mode:
        headers["X-Cls-Mode"] = cls_mode
    if detect_size > 0:
        headers["X-Detect-Resized-Width"] = str(resize_info["width"]) 
        headers["X-Detect-Resized-Height"] = str(resize_info["height"]) 
        headers["X-Detect-Scale"] = f"{resize_info['scale']:.6f}"

    return StreamingResponse(buf, media_type="image/png", headers=headers)

@app.post("/classify")
async def classify_image(
    file: UploadFile = File(..., description="Image file to classify"),
    classify_mode: str = "warp"
):
    validate_image_file(file)
    image_array, image_info = process_uploaded_image(file)
    cls = run_classification(image_array, topk=5, mode=classify_mode if classify_mode in ("warp", "crop") else "warp")
    return {
        "success": True,
        "image_info": image_info,
        "classification": cls,
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model, device
    
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "cuda_available": torch.cuda.is_available(),
        "model_info": {
            "total_classes": len(model.model.names) if model else 0,
            "model_type": "YOLOv8s" if model else "None"
        }
    }
    
    if model is None:
        status["status"] = "unhealthy"
        status["error"] = "Model not loaded"
    
    return status


@app.get("/classes")
async def get_available_classes():
    """Get list of all available object classes that can be detected"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please restart the server."
        )
    
    classes = list(model.model.names.values())
    
    return {
        "success": True,
        "message": f"Available {len(classes)} object classes",
        "total_classes": len(classes),
        "classes": classes,
        "categories": {
            "animals": [cls for cls in classes if cls in ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']],
            "vehicles": [cls for cls in classes if cls in ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']],
            "objects": [cls for cls in classes if cls not in ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']]
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI Object Detection Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
