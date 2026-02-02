import cv2 as cv
import pandas as pd

from app.loader import load_image_from_url
from app.pipeline import extract_features_from_url

DATA_PATH = "data/nvr_snapshot_history_rows.csv"
df = pd.read_csv(DATA_PATH)

BLUR_THRESHOLD = 80
DARK_THRESHOLD = 15

# ทดลองแค่ 5 แถวแรกก่อน
for _, row in df.head(2).iterrows():
    img = load_image_from_url(row["image_url"])
    features = extract_features_from_url(row["image_url"])

    if img is None or features is None:
        continue

    brightness = features["brightness"]
    blur = features["blur"]

    #-------------------------------
    # RULE Status
    #-------------------------------
    if brightness < DARK_THRESHOLD:
        status = "Dark"
        color = (0, 0, 255)  # แดง
    else:
        status = "Normal"
        color = (0, 255, 0)  # เขียว

    # -------------------------------
    # วาดกรอบ + status
    # -------------------------------
    cv.rectangle(img, (10, 10), (420, 120), color, 2)

    cv.putText(
        img,
        f"Status: {status}",
        (20, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv.putText(
        img,
        f"Brightness: {brightness:.2f}",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    cv.putText(
        img,
        f"Blur: {blur:.2f}",
        (20, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    #-------------------------------
    # แสดงรูป    
    #-------------------------------    
    cv.imshow("Camera Health Debug", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    