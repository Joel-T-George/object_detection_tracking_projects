import cv2

# Initialize video capture (use 0 for webcam or provide video path)
cap = cv2.VideoCapture('./../Video00002.mp4')  # Replace 'video.mp4' with 0 for webcam

# Read the first frame
ret, frame = cap.read()

# Select the ROI manually (drag to select the object)
roi = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)

# Initialize the tracker (using CSRT as it's good for scale variation)
tracker = cv2.TrackerCSRT_create()  # You can also use TrackerKCF_create(), TrackerMIL_create() etc.

# Initialize the tracker with the first frame and the selected ROI
tracker.init(frame, roi)

while True:
    # Capture a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker (this updates the bounding box dynamically)
    success, bbox = tracker.update(frame)

    if success:
        # Get the updated bounding box coordinates
        x, y, w, h = [int(v) for v in bbox]

        # Draw the bounding box with dynamic size
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the coordinates and size of the object
        cv2.putText(frame, f"Size: {w}x{h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # If tracking fails
        cv2.putText(frame, "Tracking Failed", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
