# Abrar AI - Professional Image Background Remover

A high-performance API for removing backgrounds from images. It features an intelligent "Auto-Detect" system that switches between Deep Learning (AI) and Precision Color Keying algorithms to ensure perfect results for both photos and logos.

## Features

* **Auto-Mode:** Automatically detects if an image is a photo or a logo/graphic on a solid background.
* **AI Mode:** Uses `isnet-general-use` for complex subjects (people, products).
* **Logo Mode:** Uses precision mathematics to preserve internal logo details (e.g., white text on white background) and applies edge feathering.
* **Custom Backgrounds:** Upload any image to automatically resize and use as the new background.
* **Solid Colors:** Easily replace backgrounds with White, Black, Green, Blue, or Transparent.

## Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure:**
    Copy the provided `.env` content into a file named `.env`.

## Running

```bash
uvicorn main:app --reload