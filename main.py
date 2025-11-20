# ========== Imports ==========
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import re
import logging
import sys
import time
import uuid
import threading
from typing import List, Optional, Tuple, Union

from fastapi import FastAPI, UploadFile, File, Request, Header, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from dotenv import load_dotenv
from rembg import remove, new_session 

# ========== Logging Configuration ==========
load_dotenv()
logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG", "false").lower() != "true" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ========== Constants ==========
ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
ALLOWED_BG_TYPES = ["transparent", "white", "black", "green", "blue"]

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 100))
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", 10))
VALID_KEYS = {os.getenv("API_KEY", "your-secret-key")}
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PROCESSED_DIR = os.path.abspath(os.getenv("PROCESSED_DIR", "processed"))

# Models & Settings
RMBG_IMAGE_MODEL = os.getenv("RMBG_IMAGE_MODEL", "isnet-general-use")

# Detection Settings
SOLID_BG_TOLERANCE = 10
COLOR_KEY_TOLERANCE = 20 

# Edge Settings (For Logo Mode)
LOGO_EDGE_FEATHER_KERNEL_SIZE = 3
LOGO_EDGE_FEATHER_SIGMA = 1.0

# Rate Limits
RATE_LIMIT_IMAGE = os.getenv("RATE_LIMIT_IMAGE", "20/minute")
RATE_LIMIT_JOBS = os.getenv("RATE_LIMIT_JOBS", "60/minute")

# Cleanup
CLEANUP_INTERVAL_SECONDS = 3600
FILE_EXPIRY_SECONDS = 86400

# Global State
jobs = {}

# ========== FastAPI App ==========
app = FastAPI(title="Abrar AI - Professional Image BG Removal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"🚨 VALIDATION ERROR: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": str(exc)}),
    )

os.makedirs(PROCESSED_DIR, exist_ok=True)
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")

# ========== Model Loading ==========
global_image_session = None
try:
    logger.info(f"Loading Image Model: {RMBG_IMAGE_MODEL}...")
    global_image_session = new_session(model_name=RMBG_IMAGE_MODEL)
    logger.info("Image Model Loaded.")
except Exception as e:
    logger.error(f"Failed to load Image Model: {e}")

# ========== Helpers ==========

def sanitize_filename(filename: str) -> str:
    name = re.sub(r'[<>:"|?*]', '_', os.path.basename(filename))
    return name[:250]

def ensure_safe_path(base, relative):
    full = os.path.abspath(os.path.join(base, relative))
    if not full.startswith(os.path.abspath(base)): raise ValueError("Path traversal")
    return full

async def verify_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if x_api_key is None or x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key

def get_bg_color_tuple(bg_type: str) -> Optional[Tuple[int, int, int]]:
    if bg_type == "white": return (255, 255, 255)
    if bg_type == "black": return (0, 0, 0)
    if bg_type == "green": return (0, 177, 64)
    if bg_type == "blue": return (0, 71, 187)
    return None

def analyze_background(img_np: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Returns (is_solid, bg_color_rgb) based on corner analysis"""
    try:
        h, w, _ = img_np.shape
        corners = [
            img_np[0, 0], img_np[0, w-1],
            img_np[h-1, 0], img_np[h-1, w-1]
        ]
        corners_np = np.array(corners)
        mean_std = np.mean(np.std(corners_np, axis=0))
        
        if mean_std < SOLID_BG_TOLERANCE:
            return True, np.mean(corners_np, axis=0).astype(np.uint8)
        return False, None
    except: return False, None

# ========== Core Logic ==========

async def process_single_image(file: UploadFile, mode: str, bg_type: str, custom_bg_pil: Optional[Image.Image] = None) -> dict:
    try:
        sanitized_name = sanitize_filename(file.filename)
        file_content = await file.read()
        
        original_img = Image.open(BytesIO(file_content)).convert("RGB")
        original_np = np.array(original_img)
        
        # 1. Determine Strategy
        method = mode
        is_solid_bg, detected_bg_color = analyze_background(original_np)
        
        if mode == "auto":
            method = "color_key" if is_solid_bg else "ai"
            
        # 2. Generate Alpha Mask (0-255)
        alpha_mask = None
        
        if method == "color_key":
            # Fallback to top-left pixel if detection was ambiguous but user forced color_key
            if detected_bg_color is None: 
                 _, detected_bg_color = analyze_background(original_np)
                 if detected_bg_color is None: detected_bg_color = original_np[0,0]

            # Calculate Bounds
            lower = np.array([max(0, int(c) - COLOR_KEY_TOLERANCE) for c in detected_bg_color], dtype=np.uint8)
            upper = np.array([min(255, int(c) + COLOR_KEY_TOLERANCE) for c in detected_bg_color], dtype=np.uint8)
            
            # Create Mask
            bg_mask_cv = await run_in_threadpool(cv2.inRange, original_np, lower, upper)
            fg_mask_cv = cv2.bitwise_not(bg_mask_cv)
            
            # Edge Feathering
            k_size = LOGO_EDGE_FEATHER_KERNEL_SIZE
            if k_size % 2 == 0: k_size += 1
            alpha_mask = await run_in_threadpool(cv2.GaussianBlur, fg_mask_cv, (k_size, k_size), LOGO_EDGE_FEATHER_SIGMA)
            
        else: # AI Mode
            if global_image_session is None: raise RuntimeError("AI Model not loaded")
            res_bytes = await run_in_threadpool(remove, file_content, session=global_image_session)
            temp_img = Image.open(BytesIO(res_bytes)).convert("RGBA")
            alpha_mask = np.array(temp_img)[:, :, 3]

        # 3. Compositing
        if alpha_mask.shape[:2] != original_np.shape[:2]:
            alpha_mask = cv2.resize(alpha_mask, (original_np.shape[1], original_np.shape[0]))
            
        final_rgba = np.dstack((original_np, alpha_mask))
        result_img = Image.fromarray(final_rgba, "RGBA")

        # --- Background Replacement ---
        if custom_bg_pil is not None:
            # Resize Custom BG to match main image
            bg_resized = custom_bg_pil.resize(result_img.size, Image.Resampling.LANCZOS).convert("RGBA")
            result_img = Image.alpha_composite(bg_resized, result_img)
            result_img = result_img.convert("RGB")
            
        elif bg_type != "transparent":
            bg_color = get_bg_color_tuple(bg_type)
            if bg_color:
                bg_layer = Image.new("RGBA", result_img.size, (*bg_color, 255))
                result_img = Image.alpha_composite(bg_layer, result_img)
                result_img = result_img.convert("RGB")

        # 4. Save
        filename = f"{uuid.uuid4().hex}.png"
        path = ensure_safe_path(PROCESSED_DIR, filename)
        await run_in_threadpool(result_img.save, path, "PNG")
        
        return {
            "success": True, 
            "file": filename, 
            "url": f"/download/image/{filename}", 
            "mode_used": method,
            "bg_type": "custom_image" if custom_bg_pil else bg_type
        }
        
    except Exception as e:
        logger.error(f"Img Error: {e}")
        return {"success": False, "error": str(e)}

# ========== Endpoints ==========

@app.post("/remove-bg")
@limiter.limit(RATE_LIMIT_IMAGE)
async def api_remove_image(
    request: Request,
    files: List[UploadFile] = File(..., description="Images to process"), 
    bg_file: Union[UploadFile, str, None] = File(default=None, description="Custom background image"),
    mode: str = Query("auto", enum=["auto", "ai", "color_key"], description="Processing Logic"),
    background_type: str = Query("transparent", enum=ALLOWED_BG_TYPES, description="Solid color fallback"),
    api_key: str = Depends(verify_key)
):
    if len(files) > MAX_FILES_PER_REQUEST: raise HTTPException(400, "Too many files")
    
    custom_bg_pil = None
    # Smart check for optional file upload
    if bg_file:
        if hasattr(bg_file, "read"):
            try:
                content = await bg_file.read()
                if len(content) > 0:
                    custom_bg_pil = Image.open(BytesIO(content)).convert("RGBA")
                    logger.info("Custom background loaded.")
            except Exception as e:
                logger.error(f"Failed to load custom background: {e}")

    results = []
    for i, f in enumerate(files):
        logger.info(f"Img {i+1}/{len(files)}: {f.filename} | Mode: {mode}")
        results.append(await process_single_image(f, mode, background_type, custom_bg_pil))
        
    return {"results": results}

@app.get("/")
def home(request: Request): return {"status": "Online", "mode": "Image Only (Universal + Custom BG)"}

@app.get("/jobs")
def jobs_ep(request: Request, api_key: str = Depends(verify_key)): return {"jobs": jobs}

# ========== Cleanup Task ==========
def background_cleanup():
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        now = time.time()
        if os.path.exists(PROCESSED_DIR):
            for f in os.listdir(PROCESSED_DIR):
                fp = os.path.join(PROCESSED_DIR, f)
                try:
                    if os.path.isfile(fp) and os.stat(fp).st_mtime < now - FILE_EXPIRY_SECONDS:
                        os.remove(fp)
                except Exception:
                    pass

threading.Thread(target=background_cleanup, daemon=True).start()