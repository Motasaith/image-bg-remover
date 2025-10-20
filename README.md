## Abrar AI – Background Removal & Video Matting API

### (Powered by RMBG 1.4 + RVM MobilenetV3 + FastAPI)

---

### Overview

**Abrar AI** is a FastAPI-based service that removes backgrounds from **images and videos** using AI models.
It combines **BRIA AI’s RMBG 1.4** model for single images and **Robust Video Matting (RVM)** for videos to produce clean, transparent outputs with various background options.

It supports:

*  Image background removal (transparent PNG output)
*  Video background matting with multiple background options (transparent, green, blue, white, black)
*  Asynchronous job queue for video processing to prevent API blocking
*  GPU acceleration (CUDA) for fast inference
*  Easy REST API access via Swagger UI (`/docs`)
*  Configurable via environment variables

---

## Features

| Feature                       | Description                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------- |
|  Image Background Removal     | Uses `briaai/RMBG-1.4` (Hugging Face) for precise segmentation.                   |
|  Video Matting                | Uses `RobustVideoMatting (RVM)` for temporal consistent video background removal. |
|  Asynchronous Job Queue       | Background processing for videos to prevent API blocking.                         |
|  Multiple Background Options  | Transparent, green screen, white, black, blue.                                    |
|  FastAPI Integration          | Exposes endpoints `/remove-bg`, `/remove-bg-video`, `/jobs` with Swagger docs.    |
|  Automatic FFmpeg Integration | Converts processed frames into `.webm` with alpha (yuva420p).                     |
|  GPU Support                  | Uses PyTorch CUDA if available, automatically switches to CPU otherwise.          |
|  Modular Design               | Replace or extend models easily.                                                  |

---

## 📁 Project Structure

```
project_root/
│
├── main.py                          # FastAPI main entrypoint
├── models/
│   └── rvm_mobilenetv3.pth          # RVM model checkpoint (download required)
│
├── processed/                       # Output directory for processed files
│
├── RobustVideoMatting/              # Cloned official RVM repo
│   ├── model.py
│   ├── inference_utils.py
│   └── ...
│
└── requirements.txt
```

---

## Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/abrar-ai-bg-removal.git
cd abrar-ai-bg-removal
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # (Linux/macOS)
venv\Scripts\activate          # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, create one containing:

```txt
fastapi
uvicorn
torch
torchvision
transformers
pillow
tqdm
opencv-python
numpy
```

### 4️⃣ Clone RobustVideoMatting

```bash
git clone https://github.com/PeterL1n/RobustVideoMatting.git
```

### 5️⃣ Download RVM Weights

Download MobilenetV3 weights (~45MB):

**URL:** [https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_mobilenetv3.pth](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_mobilenetv3.pth)

Place it in:

```
models/rvm_mobilenetv3.pth
```

### 6️⃣ Start the API

```bash
uvicorn main:app --reload
```

Access documentation and test API at:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

### 1️⃣ `/remove-bg` – Image Background Removal

**POST** – Upload an image file (`.jpg`, `.png`, `.jpeg`)
**Response:** Transparent PNG

**Example (Python):**

```python
import requests

file = {'file': open('person.jpg', 'rb')}
r = requests.post("http://127.0.0.1:8000/remove-bg", files=file)
print(r.json())
```

---

### 2️⃣ `/remove-bg-video` – Video Background Matting

**POST** – Upload `.mp4` or `.mov` file
**Optional Query:** `background_type` = `transparent`, `green`, `blue`, `white`, `black`

**Response:** Job ID for asynchronous processing

Example via `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/remove-bg-video?background_type=transparent" \
     -F "file=@input.mp4"
```

This returns a JSON with `job_id`. Poll the status using `/jobs/{job_id}`.

### 3️⃣ `/jobs` – List All Jobs

**GET** – Returns a list of all job statuses.

Example via `curl`:

```bash
curl -X GET "http://127.0.0.1:8000/jobs"
```

### 4️⃣ `/jobs/{job_id}` – Get Job Status

**GET** – Check the status of a specific job.

Example via `curl`:

```bash
curl -X GET "http://127.0.0.1:8000/jobs/123e4567-e89b-12d3-a456-426614174000"
```

When status is "completed", the response includes `result` with the download URL.

---

## How It Works

| Stage                    | Process                                                                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **1. Upload**            | The FastAPI backend accepts image/video via HTTP POST.                                                                     |
| **2. Inference**         | - For images, uses `RMBG 1.4` (deep segmentation). <br> - For videos, uses `RVM` (recurrent matting with temporal memory). |
| **3. Output Generation** | Frames are processed and saved as RGBA PNGs, then combined into `.webm` using FFmpeg.                                      |
| **4. Delivery**          | Processed file is sent back or saved in `/processed/`.                                                                     |

---

## Performance & Quality

| System Type                       | Expected Speed | Notes                                                 |
| --------------------------------- | -------------- | ----------------------------------------------------- |
| **CPU only**                      | Moderate       | Works fine for short videos, slower for long footage. |
| **GPU (CUDA)**                    | 5–10× faster   | Enables real-time or near-real-time matting.          |
| **High-end GPU (e.g., RTX 4090)** | Up to 60 FPS   | Very high-quality, stable alpha transitions.          |

**Output quality** improves slightly with better GPUs (due to FP32 precision and faster batch throughput),
but visually, the model’s performance mainly depends on training — not raw hardware.

File size decreases after processing because:

* Background pixels are transparent or uniform (compress better)
* Alpha channel encoding is efficient in VP9 (`yuva420p`)

---

## Known Limitations / Drawbacks

| Limitation              | Description                                          |
| ----------------------- | ---------------------------------------------------- |
|  No audio passthrough   | FFmpeg output drops original audio.                  |
|  Slow on CPU            | Especially for HD videos (>1080p).                   |
|  High memory usage      | RVM loads full resolution into GPU memory.           |
|  Frame-based export     | Processes each frame individually → slower encoding. |
|  No live webcam support | Only works with uploaded files.                      |
|  RVM weights required   | Model checkpoint must be manually downloaded.        |

---

## Comparison with Latest Tech (as of 2025)

| Technology                       | Model                  | Strength                             | Weakness                               |
| -------------------------------- | ---------------------- | ------------------------------------ | -------------------------------------- |
| **RVM (This Project)**           | MobilenetV3 / ResNet50 | Lightweight, fast, real-time capable | Slight edge bleeding on complex scenes |
| **RMBG 1.4 (This Project)**      | Transformer-based      | Very accurate single-image removal   | Not temporal (frame-by-frame)          |
| **SAM (Segment Anything Model)** | Meta AI                | Universal segmentation               | Overkill for background-only tasks     |
| **OmniMatte / MODNet++ (New)**   | 2024–2025 models       | Higher fidelity hair transparency    | Require large GPUs & advanced setup    |
| **RunwayML Gen-2 or Pika Labs**  | Commercial AI tools    | Best generative results              | Closed source, subscription-only       |

**Conclusion:**
RVM remains the **best open-source choice** for background matting with real-time performance on consumer GPUs.

---

## 🔧 Future Improvements

| Improvement          | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| Integrate Audio Merge| Use FFmpeg to merge processed video with original audio.      |
| Add Batch Processing | Queue multiple jobs for async processing.                     |
| Upgrade to MODNet++  | Higher fidelity on hair, shadows, and motion.                 |
| Stream Input Support | Process RTSP/webcam feed instead of saved files.              |
| Add Quality Controls | CRF and bitrate options for better compression control.       |
| Quantized Models     | Reduce model size and speed up inference on CPU-only systems. |
| Cloud Deployment     | Containerize (Docker) and deploy on AWS/GCP for scalability.  |

---

## Business & Developer Benefits

| Stakeholder    | Benefit                                                                   |
| -------------- | ------------------------------------------------------------------------- |
| **Developers** | Easily integrate background removal into apps, dashboards, or automation. |
| **Businesses** | Offer AI-powered virtual background or video-editing features.            |
| **Studios**    | Save time & cost on manual chroma key editing.                            |
| **E-commerce** | Automatically remove backgrounds for product images.                      |

---

## Example Deployment (Docker)

To containerize for production:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Then:

```bash
docker build -t abrar-ai .
docker run -p 8000:8000 abrar-ai
```

---

## License & Attribution

* **RMBG 1.4** – © BRIA AI (Apache 2.0 License)
* **RVM** – © Peter Lin (MIT License)
* **FastAPI**, **Torch**, **Transformers** – Open Source
* This integration and packaging – © 2025 Abdul Rauf Azhar (Abrar AI Project)

---

## Support or Collaboration

For collaboration, deployment help, or advanced model integration (e.g., MODNet++, Pika, or SAM 2.0),
feel free to contact or open an issue on the repository.

---
