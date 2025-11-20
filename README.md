# Abrar AI - Professional Background Remover

A production-grade API and Dashboard for removing backgrounds from **Images** and **Videos**.

Unlike standard tools that use a single AI model, Abrar AI features an **Intelligent Dual-Pipeline Engine**. It automatically detects the type of image (Photo vs. Logo) and switches between Deep Learning and Precision Color Keying to ensure perfect results for every use case.

## 🚀 Key Features

* **Auto-Detection Brain:** Instantly analyzes image content to choose the best algorithm.
* **AI Mode (Photos):** Uses `isnet-general-use` (Rembg) for complex subjects like people, cars, and animals.
* **Logo Mode (Graphics):** Uses precision math to preserve internal details (e.g., white text inside a white logo) that AI models often destroy.
* **Video Support:** Full video background removal using RVM (Robust Video Matting) with smart downsampling for low-res inputs.
* **Custom Backgrounds:** Upload any image to automatically resize and use as the new background for your subject.
* **Batch Processing:** Drag & Drop 50+ images at once; the system handles the queue automatically.

---

## 🖼️ The "Dual-Pipeline" Difference

Standard AI models often fail on logos because they treat internal white spaces as "background." Abrar AI solves this.

### 1. Complex Photos (AI Mode)
**Perfect for:** People, Products, Animals, Real-world scenes.
* **Input:** A woman holding a monkey with a complex street background.
* **Result:** The background is cleanly removed, preserving hair and fur details.

| Original | Processed |
| :---: | :---: |
| ![Original Photo](demo/org1.png) | ![Processed Photo](demo/new1.png) |

| Original | Processed |
| :---: | :---: |
| ![Original Photo](demo/org2.png) | ![Processed Photo](demo/new2.png) |


### 2. Logos & Graphics (Logo Mode)
**Perfect for:** Icons, Text, Vector graphics, Solid backgrounds.
* **Input:** A blue "ibrar ai" logo with white text on a white background.
* **Result:** The white background is removed, but the **white 'i' and text inside are preserved perfectly.**

| Original | Processed (Transparent) |
| :---: | :---: |
| ![Original Logo](demo/org.png) | ![Processed Logo](demo/new.png) |

---

## 🛠️ Installation & Local Development

### Prerequisites
* Docker & Docker Compose
* (Optional) Python 3.10+ for local dev

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/abrar-ai-backend.git](https://github.com/yourusername/abrar-ai-backend.git)
cd abrar-ai-backend
````

### 2\. Create Environment File

Create a `.env` file in the root directory:

```ini
API_KEY=super-secret-key
MAX_UPLOAD_SIZE_MB=500
ALLOWED_ORIGINS=*

# Processing Settings
RMBG_IMAGE_MODEL=isnet-general-use
SOLID_BG_TOLERANCE=10
COLOR_KEY_TOLERANCE=20

# Rate Limits
RATE_LIMIT_IMAGE=1000/minute
RATE_LIMIT_VIDEO=100/minute
```

### 3\. Run with Docker (Recommended)

This will start the API on port `8000` and the Dashboard on port `8090`.

```bash
docker compose up -d --build
```

Access the App:

  * **Dashboard:** `http://localhost:8090`
  * **API Docs:** `http://localhost:8000/docs`

-----

## ☁️ Deployment Guide (VPS)

Deploying to a DigitalOcean, AWS, or Azure VPS is simple.

### Step 1: Prepare the Server

Install Docker and Docker Compose on your VPS.

```bash
# Ubuntu Example
apt update
apt install docker.io docker-compose -y
```

### Step 2: Deploy Code

SSH into your VPS and clone the repo.

```bash
git clone [https://github.com/yourusername/abrar-ai-backend.git](https://github.com/yourusername/abrar-ai-backend.git)
cd abrar-ai-backend
```

### Step 3: Configure

1.  Create the `.env` file (copy the example above).
2.  **Important:** Edit `dashboard.html` and change `const API_URL` to your VPS Public IP.
    ```javascript
    const API_URL = "http://YOUR_VPS_IP:8000";
    ```

### Step 4: Launch

```bash
docker compose up -d --build
```

Your app is now live\!

  * **Share this link:** `http://YOUR_VPS_IP:8090`

-----

## 🔌 API Usage

You can integrate this API into any app (React, Mobile, etc.).

### Endpoint: `POST /remove-bg`

Removes background from an image.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `files` | File | The source image (Required) |
| `bg_file` | File | A custom background image to place behind the subject (Optional) |
| `mode` | String | `auto` (default), `ai`, or `color_key` |
| `background_type` | String | `transparent`, `white`, `black`, `green` (Ignored if `bg_file` is used) |

**Example (Python):**

```python
import requests

url = "http://localhost:8000/remove-bg"
files = {
    'files': open('photo.jpg', 'rb'),
    'bg_file': open('beach.jpg', 'rb') # Optional
}
headers = {'X-API-Key': 'super-secret-key'}

response = requests.post(url, files=files, headers=headers)
print(response.json())
```

### 💡 How to add the images to the README
1.  Create a folder named `docs` inside your project.
2.  Inside `docs`, create a folder named `images`.
3.  Save your comparison images there with these names:
    * `photo_original.jpg` (The monkey/woman original)
    * `photo_result.png` (The monkey/woman result)
    * `logo_original.png` (The blue logo original)
    * `logo_result.png` (The blue logo transparent result)

When you push this to GitHub, the images will show up beautifully in the README.

