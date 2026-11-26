import time

import cv2
import numpy as np

from rotoredge import config
from rotoredge.camera import Camera
from rotoredge.detectors import METHODS
from rotoredge.shapes import find_shapes, is_quad


def draw_status(img, lines):
    for i, text in enumerate(lines):
        pos = (12, 28 + i * 26)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


cam = Camera()
cv2.namedWindow(config.WINDOW)
cv2.createTrackbar("Threshold", config.WINDOW, config.DEFAULT_THRESH, 255, lambda v: None)
names = list(METHODS)
method_idx = 0
try:
    while True:
        frame = cam.read()
        if frame is None:
            break

        detect = METHODS[names[method_idx]]

        h, w = frame.shape[:2]
        scale = max(w / config.PROC_W, 1.0)
        small = cv2.resize(frame, (int(w / scale), int(h / scale)))

        thresh = cv2.getTrackbarPos("Threshold", config.WINDOW)
        t0 = time.perf_counter()
        binary, contours = find_shapes(detect(small), thresh)
        ms = (time.perf_counter() - t0) * 1000

        out = frame.copy()
        quads = 0
        for c in contours:
            cv2.drawContours(out, [(c * scale).astype(np.int32)], -1, (255, 255, 255), 3)
            quad, approx = is_quad(c)
            if quad:
                quads += 1
                approx_big = (approx * scale).astype(np.int32)
                cv2.polylines(out, [approx_big], True, (0, 255, 0), 4)

        draw_status(out, [
            f"Metode: {names[method_idx]}  (m = ganti)",
            f"Threshold: {thresh}",
            f"Kontur: {len(contours)} | Segi empat: {quads}",
            f"Proses: {ms:.1f} ms",
        ])

        cv2.imshow(config.WINDOW, out)
        cv2.imshow("Peta tepi", binary)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            method_idx = (method_idx + 1) % len(names)
        if cv2.getWindowProperty(config.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cam.release()
    cv2.destroyAllWindows()