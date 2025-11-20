# 1. Use Python 3.10 Slim (Smaller size)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install System Dependencies
# ✅ FIXED: Replaced obsolete 'libgl1-mesa-glx' with 'libgl1'
# ffmpeg -> Required for video processing
# libgl1 -> Required for OpenCV
# libglib2.0-0 -> Required for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first (Better caching)
COPY requirements.txt .

# 5. Install Python Dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your Application Code
COPY . .

# 7. Create the processed directory
RUN mkdir -p processed

# 8. Expose the Port
EXPOSE 8000

# 9. Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]