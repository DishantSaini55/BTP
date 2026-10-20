"""Sequentially verify that an extracted POC source video decodes in OpenCV."""
import sys
import cv2

path = sys.argv[1]
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise SystemExit(f"cannot open {path}")
count = 0
expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    ok, _ = cap.read()
    if not ok:
        break
    count += 1
cap.release()
print(f"decoded_frames={count} expected_frames={expected}")
if count != expected:
    raise SystemExit("sequential decode ended before the container's advertised frame count")
