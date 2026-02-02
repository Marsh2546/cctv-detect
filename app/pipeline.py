from app.loader import load_image_from_url
from app.features import mean_brightness, blur_score, texture_std


def extract_features_from_url(image_url: str):
    img = load_image_from_url(image_url)

    if img is None:
        return None

    return {
        "brightness": mean_brightness(img),
        "blur": blur_score(img),
        "texture": texture_std(img),
    }
