import cv2
import numpy as np
import time 

desired_fps = 60
frame_time = 1 / desired_fps  # Time to wait between frames in seconds 

# Initialize variables
region_selected = False
tracking = False
start_point = ()
end_point = ()
old_gray = None
p0 = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Mouse callback function to select region
def select_region(event, x, y, flags, param):
    global start_point, end_point, region_selected, tracking, p0
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        end_point = (x, y)
        region_selected = False
        tracking = False
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        region_selected = True
        tracking = False

# Create a named window and set the mouse callback function
cv2.namedWindow("Tracking Window")
cv2.setMouseCallback("Tracking Window", select_region)

# Start video capture
cap = cv2.VideoCapture("Video00002.mp4")

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Draw the selection rectangle if the region is being selected
    if region_selected and not tracking:
        x1, y1 = start_point
        x2, y2 = end_point
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Once the region is selected, initialize tracking
        if not tracking:
            # Select points within the selected region using corner detection
            mask = np.zeros_like(frame_gray)
            mask[y1:y2, x1:x2] = 255  # Mask the selected region
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7)
            if p0 is not None:
                tracking = True
                old_gray = frame_gray.copy()
    
    # If tracking, perform the tracking
    if tracking and p0 is not None:
        # Calculate the optical flow for the selected points
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Select good points where status == 1
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # Draw the new position of the points and a rectangle around the region
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
        
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    # Display the frame with tracking
    cv2.imshow("Tracking Window", frame)
    elapsed_time = time.time() - start_time 
    time_to_wait = frame_time - elapsed_time 
    if time_to_wait > 0: 
        time.sleep(time_to_wait) 
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
