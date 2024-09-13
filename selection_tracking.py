import cv2
import numpy as np


# List to store selected points
points = []
tracking = False

# Mouse callback to select points
def select_points(event, x, y, flags, param):
    global points, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        tracking = True

# Create a named window and set the mouse callback function
cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", select_points)

# Start video capture
cap = cv2.VideoCapture("/home/dhaksha/Desktop/Data/Raw_Data/Video00001.mp4")

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
old_gray = None
p0 = None

# Capture the first frame for point selection
while not tracking:
    ret, frame = cap.read()
    if not ret:
        break

    # Display frame to select points
    for point in points:
        cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)

    cv2.imshow("Select Points", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if len(points) > 0:
    # Convert selected points to the required format
    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Convert the first frame to grayscale
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Tracking motion of the selected pixels
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if p0 is not None and len(p0) > 0:
        # Calculate Optical Flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Select good points where status is 1 (successfully tracked)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # Draw the tracking lines and points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    # Display the frame with tracking
    cv2.imshow('Pixel Motion Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
