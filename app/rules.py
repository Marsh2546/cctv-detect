def decide_camera_status(features, thresholds=None):
    """
    ตัดสินสถานะกล้องจาก feature ที่วัดได้
    """

    if features is None:
        return "ERROR", (0, 0, 255)

    # -------------------------------
    # ค่า default threshold
    # -------------------------------
    default_thresholds = {
        "blur": 8838.987184,
        "dark": 88.284735,
        "texture": 43.735841
    }

    if thresholds is None:
        thresholds = default_thresholds

    brightness = features["brightness"]
    blur = features["blur"]
    texture = features.get("texture", None)

    # -------------------------------
    # Logic การตัดสิน
    # -------------------------------
    if brightness < thresholds["dark"]:
        return "Too dark.", (0, 0, 255)   # แดง

    if blur < thresholds["blur"]:
        return "BLURRY", (0, 255, 255)      # เหลือง

    if texture is not None and texture < thresholds["texture"]:
        return "BLOCKED", (255, 0, 255)     # ม่วง

    return "NORMAL", (0, 255, 0)            # เขียว

def calculate_health_score(features, thresholds=None):
    """
    คำนวณ Health Score (0–100) จาก features
    """
    status, _ = decide_camera_status(features)

    if status == "CAMERA_DOWN":
        return 0


    if features is None:
        return 0

    default_thresholds = {
        "blur": 80,
        "dark": 15,
        "texture": 15
    }

    if thresholds is None:
        thresholds = default_thresholds

    score = 100

    brightness = features["brightness"]
    blur = features["blur"]
    texture = features.get("texture", None)

    # -------------------------------
    # โทษความมืด
    # -------------------------------
    if brightness < thresholds["dark"]:
        score -= 50
    elif brightness < thresholds["dark"] * 2:
        score -= 20

    # -------------------------------
    # โทษความเบลอ
    # -------------------------------
    if blur < thresholds["blur"]:
        score -= 30
    elif blur < thresholds["blur"] * 1.5:
        score -= 15

    # -------------------------------
    # โทษการถูกบัง
    # -------------------------------
    if texture is not None :
        if texture < thresholds["texture"]:
            score -= 25
        elif texture < thresholds["texture"] * 2:
            score -= 10

    # กันไม่ให้ติดลบ
    return max(score, 0)
 
