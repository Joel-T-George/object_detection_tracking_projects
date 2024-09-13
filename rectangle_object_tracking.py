import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("/home/dhaksha/Downloads/4609535-uhd_3840_2160_24fps.mp4")

# Define the initial bounding box
initBB = None

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize some variables
old_gray = None
p0 = None

# Mouse callback function to select the region
def select_point(event, x, y, flags, param):
    global p0, old_gray, initBB
    if event == cv2.EVENT_LBUTTONDOWN:
        p0 = np.array([[x, y]], dtype=np.float32)  # Store the initial point
        initBB = (x - 50, y - 50, 100, 100)  # Define initial bounding box around the point

# Set up mouse callback
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_point)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = cv2.resize(frame, (680, 380))

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # If the point has been selected
    if p0 is not None:
        if old_gray is not None:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # If the flow is good, update the point position and the bounding box
            if st[0] == 1:
                x, y = p1.ravel()
                p0 = p1

                # Update the bounding box around the tracked point
                x1, y1 = int(x - 50), int(y - 50)
                x2, y2 = int(x + 50), int(y + 50)
                initBB = (x1, y1, 100, 100)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # You can segment the object inside the box (e.g., by color or other techniques)
                # For demonstration, we'll extract the region of interest (ROI)
                roi = frame[y1:y2, x1:x2]
                # Perform segmentation or any operation here on the ROI

        old_gray = frame_gray.copy()

    # Display the frame
    cv2.imshow('Tracking', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
