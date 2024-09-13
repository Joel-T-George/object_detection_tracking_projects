import cv2
import numpy as np

# Initialize variables
region_selected = False
tracking = False
paused = False
start_point = ()
end_point = ()
old_gray = None
p0 = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Mouse callback function to select region
def select_region(event, x, y, flags, param):
    global start_point, end_point, region_selected
    if paused:
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            end_point = (x, y)
            region_selected = False
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            region_selected = True

# Create a named window and set the mouse callback function
cv2.namedWindow("Tracking Window")
cv2.setMouseCallback("Tracking Window", select_region)

# Start video capture
cap = cv2.VideoCapture("/home/dhaksha/Desktop/detection-yolo/Video00002.mp4")

index = 0
while True:
    # Get next frame
    ok, frame = cap.read()

    if not ok:
        print("[ERROR] reached end of file")
        break

    if index == 0:
        frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Initialize the HSV canvas with the same shape as the frame
        hsv_canvas = np.zeros_like(frame)
        hsv_canvas[..., 1] = 255  # Saturation is set to maximum
        index += 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compare initial frame with current frame
    flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    # Get x and y coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set hue of HSV canvas (position 0, which is hue)
    hsv_canvas[..., 0] = angle * (180 / np.pi) / 2

    # Set pixel intensity value (position 2, which is value)
    hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to RGB to display
    frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)

    cv2.imshow('Optical Flow (dense)', frame_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set initial frame to current frame
    frame_gray_init = frame_gray

cap.release()
cv2.destroyAllWindows()
