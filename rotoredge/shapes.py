import cv2
import numpy as np
from . import config


def find_shapes(edges, thresh):
    _, binary = cv2.threshold(edges, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = config.MIN_AREA_RATIO * edges.size
    return binary, [c for c in contours if cv2.contourArea(c) > min_area]


def is_quad(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return len(approx) == 4 and cv2.isContourConvex(approx), approx