import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from app.pipeline import extract_features_from_url

# -------------------------------
# โหลดข้อมูล
# -------------------------------
DATA_PATH = "data/nvr_snapshot_history_rows.csv"
  
df = pd.read_csv(DATA_PATH)
df = df[df["image_url"].str.startswith("http", na=False)]

# -------------------------------
# เตรียม list เก็บค่า
# -------------------------------
blur_values = []
brightness_values = []
texture_values = []

# -------------------------------
# loop เก็บ feature
# -------------------------------
for row in tqdm(
    df.itertuples(index=False),
    total=len(df),
    desc="Analyzing distribution"
):

    features = extract_features_from_url(row.image_url)
    if features is None:
        continue

    blur_values.append(features["blur"])
    brightness_values.append(features["brightness"])
    texture_values.append(features["texture"])
    

# -------------------------------
# วิเคราะห์ตัวเลข
# -------------------------------
df_stats = pd.DataFrame({
    "blur": blur_values,
    "brightness": brightness_values,
    "texture": texture_values
})
# print("Collected sample:", len(df_stats))

print(df_stats.describe())

print("\nPercentiles:")
print("Blur p10:", df_stats["blur"].quantile(0.10))
print("Brightness p10:", df_stats["brightness"].quantile(0.10))
print("Texture p10:", df_stats["texture"].quantile(0.10))

# -------------------------------
def compute_median_thresholds(stats_df: pd.DataFrame, p=0.10) -> dict:
    """
    หาค่ากลาง (median) ของแต่ละตัวสำหรับใช้เป็น default_thresholds
    """
    return {
        "blur": float(stats_df["blur"].quantile(p)),
        "dark": float(stats_df["brightness"].quantile(p)),
        "texture": float(stats_df["texture"].quantile(p)),
    }


default_thresholds = compute_median_thresholds(df_stats, p=0.10)
print("\nMedian default_thresholds (p10):")
print(default_thresholds)
