import cv2
import numpy as np
import time

# Global variables
roi_defined = False
roi = None
prev_gray = None
prev_points = None
frame_time = 1/30

# Mouse callback function to set ROI
def mouse_callback(event, x, y, flags, param):
    global roi_defined, roi, prev_gray, prev_points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at ({x}, {y})")
        size = 50
        x1, y1 = max(x - size, 0), max(y - size, 0)
        x2, y2 = x + size, y + size
        roi = (x1, y1, x2, y2)
        roi_defined = True
        
        # Initialize optical flow points
        if prev_gray is not None:
            prev_points = np.array([[x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]], dtype=np.float32)

# Function to process each frame
def process_frame(frame, roi_defined, roi, prev_points):
    global prev_gray
    if roi_defined and roi is not None:
        x1, y1, x2, y2 = roi
        
        # Extract ROI and convert to grayscale
        roi_frame = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:
            prev_gray = gray_roi
            return frame, prev_points
        
        # Calculate optical flow
        if prev_points is not None and len(prev_points) > 0:
            # Calculate optical flow
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_roi, prev_points, None)
            
            # Check if optical flow calculation was successful
            if next_points is not None and status is not None:
                # Flatten status array
                status = status.flatten()
                good_new = next_points[status == 1]
                
                # Draw the points on the frame
                for pt in good_new:
                    cv2.circle(frame, (int(pt[0] + x1), int(pt[1] + y1)), 5, (0, 255, 0), -1)
                
                # Update ROI based on the new points
                if len(good_new) > 0:
                    x_coords, y_coords = zip(*good_new)
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    cv2.rectangle(frame, (int(x_min + x1), int(y_min + y1)), (int(x_max + x1), int(y_max + y1)), (255, 0, 0), 2)
        
        # Update previous frames and points
        prev_gray = gray_roi
        if roi is not None:
            x1, y1, x2, y2 = roi
            prev_points = np.array([[x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]], dtype=np.float32)
    
    return frame, prev_points

# Main function to capture and process video
def main():
    global prev_gray, prev_points
    
    # Open video capture
    cap = cv2.VideoCapture("Video00002.mp4")  # Change the index if needed, e.g., 1 for an external webcam
    
    # Set mouse callback function
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        start_time = time.time()
        if not ret:
            break
        
        # Process the frame
        frame, prev_points = process_frame(frame, roi_defined, roi, prev_points)
        
        # Draw the initial ROI rectangle if defined
        if roi_defined and roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display results
        cv2.imshow('Video', frame)
        elasped_time = start_time - time.time()
        wait_time = frame_time -elasped_time
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
