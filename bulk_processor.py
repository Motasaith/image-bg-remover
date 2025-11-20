import requests
import os

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
API_KEY = "super-secret-key"

# Define your batch here
# Format: {"file": "path/to/image.jpg", "bg": "path/to/bg.png" OR None}
tasks = [
    {"file": "test_images/photo1.jpg", "bg": "backgrounds/beach.jpg"}, # Custom BG
    {"file": "test_images/logo.png",   "bg": None},                    # Transparent
    {"file": "test_images/car.jpg",    "bg": "backgrounds/road.jpg"},  # Custom BG
]

def process_image(image_path, bg_path=None):
    url = f"{API_URL}/remove-bg"
    headers = {"X-API-Key": API_KEY}
    
    # Prepare files
    files = {'files': open(image_path, 'rb')}
    if bg_path:
        files['bg_file'] = open(bg_path, 'rb')
        
    params = {
        "mode": "auto",
        "background_type": "transparent" # Default if no bg_file
    }

    print(f"Processing: {image_path} ...")
    try:
        response = requests.post(url, headers=headers, files=files, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Download: {API_URL}{data['results'][0]['url']}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    finally:
        # Close files
        files['files'].close()
        if 'bg_file' in files: files['bg_file'].close()

if __name__ == "__main__":
    print(f"Starting batch of {len(tasks)} images...\n")
    for task in tasks:
        if os.path.exists(task['file']):
            process_image(task['file'], task.get('bg'))
        else:
            print(f"⚠️ File not found: {task['file']}")