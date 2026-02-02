import cv2
import numpy as np


def mean_brightness(img):
    """
    ค่าเฉลี่ยความสว่างของภาพ
    """
    return float(np.mean(img))


def blur_score(img):
    """
    ใช้ Laplacian Variance วัดความเบลอ
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(score)


def texture_std(img):
    """
    วัดความแปรปรวนของ texture (ใช้ detect การบังเลนส์)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.std())
