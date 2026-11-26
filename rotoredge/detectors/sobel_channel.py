import cv2
import numpy as np


def detect(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    mags = []
    for ch in cv2.split(blurred):
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mags.append(cv2.magnitude(gx, gy))
    return cv2.convertScaleAbs(np.maximum.reduce(mags))