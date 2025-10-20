 # main.py
from fastapi import FastAPI, UploadFile, File, Request, Header, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import shutil
import tempfile
import subprocess

import sys
import time
import datetime
import filetype
from tqdm import tqdm
from torchvision import transforms
from transformers import pipeline
import threading
import signal
import uuid

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== Constants ==========
ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTS = {"mp4", "mov", "avi", "mkv"}
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 100))
VALID_KEYS = {os.getenv("API_KEY", "your-secret-key")}
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
RMBG_MODEL = os.getenv("RMBG_MODEL", "briaai/RMBG-1.4")
RVM_MODEL = os.getenv("RVM_MODEL", "mobilenetv3")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed")
RATE_LIMIT_IMAGE = os.getenv("RATE_LIMIT_IMAGE", "10/minute")
RATE_LIMIT_VIDEO = os.getenv("RATE_LIMIT_VIDEO", "5/minute")
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", 3600))
FILE_EXPIRY_SECONDS = int(os.getenv("FILE_EXPIRY_SECONDS", 86400))
FFMPEG_CODEC = os.getenv("FFMPEG_CODEC", "libvpx-vp9")
FFMPEG_CRF = os.getenv("FFMPEG_CRF", "30")
FFMPEG_BV = os.getenv("FFMPEG_BV", "0")

# ========== FastAPI App ==========
app = FastAPI(title="Abrar AI - RMBG 1.4 + RVM + NAFNet Sharpening")

# Add CORS middleware
allow_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Mount static files for downloads
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device for PyTorch: {device}")

# ========== Global Exception Handler ==========
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url)},
    )

# ========== Request Logging Middleware ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.datetime.now()
    response = await call_next(request)
    duration = (datetime.datetime.now() - start).total_seconds()
    if DEBUG:
        print(f"[{datetime.datetime.now()}] {request.method} {request.url.path} -> {response.status_code} ({duration:.2f}s)")
    return response

# ========== API Key Verification ==========
async def verify_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ========== Input Validation ==========
def validate_upload(file: UploadFile, file_type="image"):
    # Limit size
    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_SIZE_MB} MB allowed.")

    ext = file.filename.split(".")[-1].lower()
    if file_type == "image" and ext not in ALLOWED_IMAGE_EXTS:
        raise ValueError("Invalid image format. Only png, jpg, jpeg supported.")
    if file_type == "video" and ext not in ALLOWED_VIDEO_EXTS:
        raise ValueError("Invalid video format. Only mp4, mov, avi, mkv supported.")

    # Basic validation
    if file_type == "image":
        kind = filetype.guess(file.file.read(512))
        file.file.seek(0)
        if not kind or kind.extension not in ALLOWED_IMAGE_EXTS:
            raise ValueError("File is not a valid image.")

# ========== Load RMBG Model (image) ==========
image_model = pipeline(
    "image-segmentation",
    model=RMBG_MODEL,
    trust_remote_code=True,
    device=0 if device == "cuda" else -1
)

@app.post("/remove-bg")
@limiter.limit("10/minute")
async def remove_bg(request: Request, file: UploadFile = File(...), _: str = Depends(verify_key)):
    try:
        validate_upload(file, "image")
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        result = image_model(image)
        arr = np.array(result)
        # assume the segmentation output supports RGBA
        if arr.shape[-1] == 4:
            from PIL import ImageFilter
            alpha = arr[:, :, 3]
            smooth = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(1.5))
            arr[:, :, 3] = np.array(smooth)
        result_img = Image.fromarray(arr)
        os.makedirs("processed", exist_ok=True)
        filename = f"{file.filename.rsplit('.',1)[0]}_no_bg.png"
        path = os.path.join("processed", filename)
        result_img.save(path)
        return {"message": "✅ Background removed!", "download_url": f"/processed/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# ========== Video Background Removal (RVM) ==========
try:
    repo_path = os.path.join(os.path.dirname(__file__), "RobustVideoMatting")
    if os.path.isdir(repo_path):
        sys.path.insert(0, repo_path)
    from RobustVideoMatting.model import MattingNetwork
except Exception as e:
    MattingNetwork = None
    _rvm_import_error = e

RVM_WEIGHTS = os.getenv("RVM_WEIGHTS_PATH", "models/rvm_mobilenetv3.pth")

def ffmpeg_exists():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

@app.post("/remove-bg-video")
@limiter.limit("5/minute")
async def remove_bg_video(request: Request, file: UploadFile = File(...), background_type: str = "transparent", _: str = Depends(verify_key)):
    # Remove background via RVM
    if MattingNetwork is None:
        return JSONResponse({"error": "RVM MattingNetwork not found.", "detail": str(_rvm_import_error)})
    if not os.path.isfile(RVM_WEIGHTS):
        return JSONResponse({"error": "RVM weights missing.", "fix": "Place models/rvm_mobilenetv3.pth in models folder."})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    os.makedirs("processed", exist_ok=True)
    base = os.path.splitext(file.filename)[0]
    output_dir = os.path.join("processed", f"{base}_rvm")
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames_rgba")
    os.makedirs(frames_dir, exist_ok=True)

    model = MattingNetwork('mobilenetv3').eval().to(device)
    model.load_state_dict(torch.load(RVM_WEIGHTS, map_location=device))

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    rec = [None]*4
    transform = transforms.ToTensor()

    print(f"🎞️ Processing {total} frames at {fps} fps ({width}x{height})...")

    for i in tqdm(range(total), desc="Matting"):
        ret, frame = cap.read()
        if not ret:
            break
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        src = transform(pil_frame).unsqueeze(0).to(device, torch.float32)

        with torch.no_grad():
            fgr, pha, *rec = model(src, *rec)

        # Keep tensors on GPU for compositing
        fgr = fgr[0].permute(1,2,0)  # [H, W, 3]
        pha = pha[0,0]  # [H, W]

        if background_type == "green":
            bg = torch.tensor([0,1,0], dtype=torch.float32, device=device)
        elif background_type == "blue":
            bg = torch.tensor([0,0,1], dtype=torch.float32, device=device)
        elif background_type == "white":
            bg = torch.tensor([1,1,1], dtype=torch.float32, device=device)
        elif background_type == "black":
            bg = torch.tensor([0,0,0], dtype=torch.float32, device=device)
        else:
            bg = None

        if bg is not None:
            # Compositing on GPU
            comp = fgr * pha.unsqueeze(-1) + bg * (1 - pha.unsqueeze(-1))
            rgba_tensor = torch.cat([comp, pha.unsqueeze(-1)], dim=-1)  # [H, W, 4]
        else:
            # Transparent background
            rgba_tensor = torch.cat([fgr * pha.unsqueeze(-1), pha.unsqueeze(-1)], dim=-1)

        # Move to CPU for PIL
        rgba_np = (rgba_tensor * 255).clamp(0, 255).byte().cpu().numpy()
        frame_out = Image.fromarray(rgba_np, "RGBA")



        out_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
        frame_out.save(out_path)

    cap.release()

    # Encode video
    webm_path = os.path.join("processed", f"{base}_rvm.webm")

    if not ffmpeg_exists():
        return JSONResponse({"message": "ffmpeg missing", "frames_dir": frames_dir})

    encode_cmd = ["ffmpeg","-y","-framerate", str(fps),
                  "-i", os.path.join(frames_dir, "frame_%06d.png"),
                  "-c:v", FFMPEG_CODEC, "-pix_fmt","yuva420p", "-crf", FFMPEG_CRF, "-b:v", FFMPEG_BV, webm_path]
    subprocess.run(encode_cmd, check=True)

    # Audio passthrough
    audio_temp = os.path.join(output_dir, "audio.aac")
    extract_audio = ["ffmpeg","-y","-i", input_path, "-vn", "-acodec","copy", audio_temp]
    merge_path = os.path.join("processed", f"{base}_rvm_with_audio.webm")

    try:
        subprocess.run(extract_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        merge = ["ffmpeg","-y","-i", webm_path,
                 "-i", audio_temp, "-c:v","copy","-c:a","libopus",
                 "-map","0:v:0","-map","1:a:0", merge_path]
        subprocess.run(merge, check=True)
        os.remove(webm_path)
        webm_path = merge_path
    except subprocess.CalledProcessError:
        print("⚠️ Audio passthrough failed, video will be silent.")

    shutil.rmtree(frames_dir)
    os.remove(input_path)
    if os.path.exists(audio_temp):
        os.remove(audio_temp)
    shutil.rmtree(output_dir)

    return FileResponse(webm_path, media_type="video/webm", filename=os.path.basename(webm_path))





# ========== Job Queue ==========
jobs = {}

def process_video_background_removal(job_id: str, input_path: str, background_type: str):
    try:
        jobs[job_id]["status"] = "processing"

        os.makedirs("processed", exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join("processed", f"{base}_rvm")
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames_rgba")
        os.makedirs(frames_dir, exist_ok=True)

        model = MattingNetwork('mobilenetv3').eval().to(device)
        model.load_state_dict(torch.load(RVM_WEIGHTS, map_location=device))

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

        rec = [None]*4
        transform = transforms.ToTensor()

        print(f"🎞️ Processing {total} frames at {fps} fps ({width}x{height})...")

        for i in tqdm(range(total), desc="Matting"):
            ret, frame = cap.read()
            if not ret:
                break
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            src = transform(pil_frame).unsqueeze(0).to(device, torch.float32)

            with torch.no_grad():
                fgr, pha, *rec = model(src, *rec)

            # Keep tensors on GPU for compositing
            fgr = fgr[0].permute(1,2,0)  # [H, W, 3]
            pha = pha[0,0]  # [H, W]

            if background_type == "green":
                bg = torch.tensor([0,1,0], dtype=torch.float32, device=device)
            elif background_type == "blue":
                bg = torch.tensor([0,0,1], dtype=torch.float32, device=device)
            elif background_type == "white":
                bg = torch.tensor([1,1,1], dtype=torch.float32, device=device)
            elif background_type == "black":
                bg = torch.tensor([0,0,0], dtype=torch.float32, device=device)
            else:
                bg = None

            if bg is not None:
                # Compositing on GPU
                comp = fgr * pha.unsqueeze(-1) + bg * (1 - pha.unsqueeze(-1))
                rgba_tensor = torch.cat([comp, pha.unsqueeze(-1)], dim=-1)  # [H, W, 4]
            else:
                # Transparent background
                rgba_tensor = torch.cat([fgr * pha.unsqueeze(-1), pha.unsqueeze(-1)], dim=-1)

            # Move to CPU for PIL
            rgba_np = (rgba_tensor * 255).clamp(0, 255).byte().cpu().numpy()
            frame_out = Image.fromarray(rgba_np, "RGBA")

            out_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            frame_out.save(out_path)

        cap.release()

        # Encode video
        webm_path = os.path.join("processed", f"{base}_rvm.webm")

        if not ffmpeg_exists():
            jobs[job_id] = {"status": "failed", "error": "ffmpeg missing"}
            return

        encode_cmd = ["ffmpeg","-y","-framerate", str(fps),
                      "-i", os.path.join(frames_dir, "frame_%06d.png"),
                      "-c:v", FFMPEG_CODEC, "-pix_fmt","yuva420p", "-crf", FFMPEG_CRF, "-b:v", FFMPEG_BV, webm_path]
        subprocess.run(encode_cmd, check=True)

        # Audio passthrough
        audio_temp = os.path.join(output_dir, "audio.aac")
        extract_audio = ["ffmpeg","-y","-i", input_path, "-vn", "-acodec","copy", audio_temp]
        merge_path = os.path.join("processed", f"{base}_rvm_with_audio.webm")

        try:
            subprocess.run(extract_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            merge = ["ffmpeg","-y","-i", webm_path,
                     "-i", audio_temp, "-c:v","copy","-c:a","libopus",
                     "-map","0:v:0","-map","1:a:0", merge_path]
            subprocess.run(merge, check=True)
            os.remove(webm_path)
            webm_path = merge_path
        except subprocess.CalledProcessError:
            print("⚠️ Audio passthrough failed, video will be silent.")

        shutil.rmtree(frames_dir)
        os.remove(input_path)
        if os.path.exists(audio_temp):
            os.remove(audio_temp)
        shutil.rmtree(output_dir)

        jobs[job_id] = {"status": "completed", "result": f"/processed/{os.path.basename(webm_path)}"}

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}

@app.post("/remove-bg-video")
@limiter.limit("5/minute")
async def remove_bg_video(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), background_type: str = "transparent", _: str = Depends(verify_key)):
    # Check for RVM availability
    if MattingNetwork is None:
        return JSONResponse({"error": "RVM MattingNetwork not found.", "detail": str(_rvm_import_error)})
    if not os.path.isfile(RVM_WEIGHTS):
        return JSONResponse({"error": "RVM weights missing.", "fix": "Place models/rvm_mobilenetv3.pth in models folder."})

    try:
        validate_upload(file, "video")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    background_tasks.add_task(process_video_background_removal, job_id, input_path, background_type)

    return {"job_id": job_id, "status": "pending", "message": "Video background removal job submitted."}

@app.get("/jobs")
def list_jobs():
    return {"jobs": jobs}

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

# ========== Auto Cleanup ==========
def cleanup_old_files():
    while True:
        time.sleep(3600)  # Run every hour
        now = time.time()
        for root, dirs, files in os.walk("processed"):
            for file in files:
                file_path = os.path.join(root, file)
                if now - os.path.getmtime(file_path) > 86400:  # 24 hours
                    os.remove(file_path)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

@app.get("/")
def home():
    return {
        "message": "Abrar AI - RMBG 1.4 + RVM",
        "routes": ["/remove-bg", "/remove-bg-video", "/jobs"],
        "note": "Use /remove-bg-video for video background removal. Check /jobs for async job status."
    }
