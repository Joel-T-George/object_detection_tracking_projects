import cv2
import numpy as np
import time

# Initialize variables
point_selected = False
point = ()
p0 = None
old_gray = None

desired_fps = 60
frame_time = 1 / desired_fps  # Time to wait between frames in seconds 
 

# Mouse callback to select a point
def select_point(event, x, y, flags, param):
    global point, point_selected, p0
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        p0 = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

# Create a named window and set the mouse callback function
cv2.namedWindow("Tracking Window")
cv2.setMouseCallback("Tracking Window", select_point)

# Start video captu
cap = cv2.VideoCapture("/home/dhaksha/Downloads/4609535-uhd_3840_2160_24fps.mp4")

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Process the video stream
while cap.isOpened():
    start_time = time.time() 
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (680, 480))
    
    # Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if point_selected:
        # Track the point using Lucas-Kanade Optical Flow
        if old_gray is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # If the point is successfully tracked (status == 1)
            if st[0] == 1:
                # Update the point location and draw the circle
                new_point = p1[0].ravel()
                cv2.circle(frame, (int(new_point[0]), int(new_point[1])), 5, (0, 0, 255), -1)
                p0 = p1
        
        # Draw the initial selected point
        #cv2.circle(frame, point, 5, (0, 255, 0), 2)
    
    # Update the previous frame
    old_gray = frame_gray.copy()

    # Display the frame in the same window
    cv2.imshow("Tracking Window", frame)

    elapsed_time = time.time() - start_time 
    time_to_wait = frame_time - elapsed_time 
    if time_to_wait > 0: 
        time.sleep(time_to_wait) 
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
