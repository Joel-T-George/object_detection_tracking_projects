import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("/home/dhaksha/Desktop/detection-yolo/Video00002.mp4")

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize some variables
old_gray = None
p0 = None
roi = None

# Mouse callback function to select the region
def select_point(event, x, y, flags, param):
    global p0, old_gray, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        p0 = np.array([[x, y]], dtype=np.float32)  # Store the initial point

        # Define ROI around the selected point
        x1, y1 = x - 50, y - 50
        x2, y2 = x + 50, y + 50
        roi = (x1, y1, x2, y2)  # ROI coordinates

        # Draw initial bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)

# Set up mouse callback
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_point)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # If the point has been selected and ROI is defined
    if p0 is not None and roi is not None:
        if old_gray is not None:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # If the flow is good, update the point position
            if st[0] == 1:
                x, y = p1.ravel()
                p0 = p1

                # Update the ROI around the tracked point
                x1, y1, x2, y2 = roi
                x1, y1 = int(x - 50), int(y - 50)
                x2, y2 = int(x + 50), int(y + 50)
                roi = (x1, y1, x2, y2)

                # Draw updated bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract the region of interest (ROI)
                if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                    roi_img = frame[y1:y2, x1:x2]

                    # Perform additional object detection or processing here on roi_img
                    # For example, convert to grayscale and apply a binary threshold
                    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                    _, binary_roi = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY)
                    cv2.imshow('ROI', binary_roi)

        old_gray = frame_gray.copy()

    # Display the frame with bounding box
    cv2.imshow('Tracking', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
