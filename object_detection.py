import cv2
import time
from pathlib import Path
from picamera2 import Picamera2

# Model files
MODEL_DIR = Path(__file__).resolve().parent / "models"

net = cv2.dnn.readNetFromCaffe(
    str(MODEL_DIR / "MobileNetSSD_deploy.prototxt"),
    str(MODEL_DIR / "MobileNetSSD_deploy.caffemodel")
)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 3

# Camera setup
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 60},
    buffer_count=4
)

picam2.configure(config)
picam2.start()

frame_count = 0
detections_cache = []
prev_time = time.perf_counter()

try:
    while True:
        frame = picam2.capture_array()

        # RGB888 capture is BGR-compatible for OpenCV
        frame = cv2.flip(frame, -1)

        h, w = frame.shape[:2]
        frame_count += 1

        # Perform detection every 3 frames
        if frame_count % DETECTION_INTERVAL == 1:
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=0.007843,
                size=(300, 300),
                mean=(127.5, 127.5, 127.5),
                swapRB=False,
                crop=False
            )

            net.setInput(blob)
            results = net.forward()
            detections_cache = []

            for i in range(results.shape[2]):
                confidence = float(results[0, 0, i, 2])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(results[0, 0, i, 1])

                if not 1 <= class_id < len(CLASSES):
                    continue

                coords = results[0, 0, i, 3:7]
                x1 = max(0, min(w - 1, int(coords[0] * w)))
                y1 = max(0, min(h - 1, int(coords[1] * h)))
                x2 = max(0, min(w - 1, int(coords[2] * w)))
                y2 = max(0, min(h - 1, int(coords[3] * h)))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections_cache.append(
                    (x1, y1, x2, y2, CLASSES[class_id], confidence)
                )

        # Draw detections
        for x1, y1, x2, y2, label, confidence in detections_cache:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"{label}: {confidence:.0%}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        # FPS measurement
        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}  Objects: {len(detections_cache)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
