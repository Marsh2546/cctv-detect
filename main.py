import cv2 as cv
import pandas as pd
from tqdm import tqdm

from app.loader import load_image_from_url
from app.pipeline import extract_features_from_url
from app.rules import decide_camera_status, calculate_health_score


# ===============================
# CONFIG
# ===============================
DATA_PATH = "data/nvr_snapshot_history_rows.csv"

MAX_SAMPLES = 300          # None = ใช้ทั้งหมด | แนะนำ 200–500 ตอน dev
SHOW_ONLY_ABNORMAL = True  # แสดงเฉพาะกล้องที่ผิดปกติ
SHOW_DEBUG = True          # แสดงภาพหรือไม่
DEBUG_WAIT_MS = 300        # เวลาที่ภาพแสดง (ms), 0 = กดเอง


def main():
    # -------------------------------
    # 1) Load CSV
    # -------------------------------
    df = pd.read_csv(DATA_PATH)

    # กรอง URL ที่ใช้ไม่ได้
    df = df[df["image_url"].str.startswith("http", na=False)]

    if MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)

    print(f"📄 Loaded rows: {len(df)}")

    if df.empty:
        print("⛔ No valid image_url found")
        return

    # -------------------------------
    # 2) Phase 1: Collect & Compute
    # -------------------------------
    results = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Processing images"
    ):
        image_url = row.image_url
        camera_name = row.camera_name

        features = extract_features_from_url(image_url)
        if features is None:
            continue

        status, _ = decide_camera_status(features)
        health_score = calculate_health_score(features)

        results.append({
            "camera_name": camera_name,
            "image_url": image_url,
            "blur": features["blur"],
            "brightness": features["brightness"],
            "texture": features["texture"],
            "status": status,
            "health_score": health_score
        })

    print(f"✅ Computed features: {len(results)} images")

    if not results:
        print("⛔ No valid data to analyze")
        return

    # -------------------------------
    # 3) Phase 2: Analyze (เร็ว)
    # -------------------------------
    df_results = pd.DataFrame(results)

    print("\n=== SUMMARY STATISTICS ===")
    print(df_results[["blur", "brightness", "texture"]].describe())

    print("\n=== STATUS COUNT ===")
    print(df_results["status"].value_counts())

    # -------------------------------
    # 4) Phase 3: Display (ทีเดียว)
    # -------------------------------
    if not SHOW_DEBUG:
        return

    if SHOW_ONLY_ABNORMAL:
        display_df = df_results[df_results["status"] != "NORMAL"]
        print(f"\n🚨 Abnormal cameras: {len(display_df)}")
    else:
        display_df = df_results

    for row in display_df.itertuples(index=False):
        img = load_image_from_url(row.image_url)
        if img is None:
            continue

        status = row.status
        score = row.health_score

        color = (0, 0, 255) if status != "NORMAL" else (0, 255, 0)

        cv.rectangle(img, (10, 10), (420, 160), color, 2)

        cv.putText(
            img,
            f"Camera: {row.camera_name}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv.putText(
            img,
            f"STATUS: {status}",
            (20, 75),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        cv.putText(
            img,
            f"Health Score: {score}",
            (20, 115),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        cv.imshow("Camera Health Result", img)
        cv.waitKey(DEBUG_WAIT_MS)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
