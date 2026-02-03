import cv2 as cv
import pandas as pd

from app.loader import load_image_from_url
from app.pipeline import extract_features_from_url
from app.rules import decide_camera_status, calculate_health_score

# -------------------------------
# โหลดข้อมูล
# -------------------------------
DATA_PATH = "data/nvr_snapshot_history_rows.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------
# เลือกดูแค่ไม่กี่ภาพ
# -------------------------------
for _, row in df.head(5).iterrows():
    img = load_image_from_url(row["image_url"])
    if img is None:
        continue

    features = extract_features_from_url(row["image_url"])
    if features is None:
        continue

    status, color = decide_camera_status(features)
    health_score = calculate_health_score(features)

    # -------------------------------
    # วาดผลลัพธ์
    # -------------------------------
    cv.rectangle(img, (10, 10), (420, 160), color, 2)

    cv.putText(img, f"STATUS: {status}", (20, 45),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv.putText(img, f"Brightness: {features['brightness']:.2f}", (20, 75),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv.putText(img, f"Blur: {features['blur']:.2f}", (20, 100),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv.putText(img, f"Health Score: {health_score}", (20, 140),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv.imshow("Camera Debug", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
