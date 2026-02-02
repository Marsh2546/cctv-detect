import requests
import cv2
import numpy as np


def load_image_from_url(url: str, timeout: int = 5):
    """
    Load image from URL and return OpenCV image (BGR)
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        image_bytes = np.asarray(
            bytearray(response.content), dtype=np.uint8
        )
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        print("กำลังโหลดรูปภาพจาก: ", url)
        return img

    except Exception as e:
        print(f"[ERROR] load_image_from_url: {e}")
        return None
