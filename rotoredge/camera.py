import cv2
from . import config


class Camera:
    def __init__(self, index=config.CAM_INDEX, width=config.WIDTH, height=config.HEIGHT):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError("Kamera tidak bisa dibuka.")

    def read(self):
        ok, frame = self.cap.read()
        return cv2.flip(frame, 1) if ok else None

    def release(self):
        self.cap.release()