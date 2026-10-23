import cv2
import time
from picamera2 import Picamera2

# Initialize Raspberry Pi Camera Module
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (1024, 576), "format": "RGB888"},
    controls={"FrameRate": 30}
)

picam2.configure(config)
picam2.start()

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

mode = "normal"
prev_time = time.time()

save_fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "output.avi",
    fourcc,
    save_fps,
    (1024, 576)
)

while True:

    # Capture from Pi Camera Module
    frame = picam2.capture_array()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    if mode == "threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, display_frame = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )
        display_frame = cv2.cvtColor(
            display_frame, cv2.COLOR_GRAY2BGR
        )

    elif mode == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        display_frame = cv2.cvtColor(
            edges, cv2.COLOR_GRAY2BGR
        )

    elif mode == "bg_sub":
        fg_mask = bg_subtractor.apply(frame)
        display_frame = cv2.bitwise_and(
            frame, frame, mask=fg_mask
        )

    elif mode == "contour":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        display_frame = cv2.drawContours(
            frame.copy(),
            contours,
            -1,
            (0, 255, 0),
            2
        )

    curr_time = time.time()
    processing_fps = 1 / max(curr_time - prev_time, 0.000001)
    prev_time = curr_time

    cv2.putText(
        display_frame,
        f"FPS: {int(processing_fps)} Mode: {mode}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    out.write(display_frame)

    cv2.imshow("Live Video", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("t"):
        mode = "threshold"
    elif key == ord("e"):
        mode = "edge"
    elif key == ord("b"):
        mode = "bg_sub"
    elif key == ord("c"):
        mode = "contour"
    elif key == ord("n"):
        mode = "normal"
    elif key == ord("q"):
        break

picam2.stop()
out.release()
cv2.destroyAllWindows()