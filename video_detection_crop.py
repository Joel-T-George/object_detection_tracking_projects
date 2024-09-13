
import cv2
import os
import torch


cap = cv2.VideoCapture("/home/dhaksha/Desktop/detection-yolo/19-august-24-model-9.mp4")
#model =yolov5.load("/home/dhaksha/Desktop/detection-yolo/yolov5/runs/yolov5s_results2/weights/best.pt")
model = torch.hub.load('ultralytics/yolov5','custom',path='/home/dhaksha/Desktop/detection-yolo/yolov5/runs/yolov5s_results2/weights/best.pt')
model.to('cpu').eval()
# Perform object detection on the frame
    #results = model(frame)

def preprocess_image(frame):
    img_resized = cv2.resize(frame,(640,640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_rgb).float().to("cpu")
    img_tensor /= 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) 
    return img_tensor

def detect_and_draw_boxes(frame,crop_output_dir,last_detection):
    orginal_height , orginal_width = frame.shape[:2]
    img_tensor = preprocess_image(frame)
    results= model(frame)
    labels =results.names
    print(labels)
    boxes = results.xyxy[0].cpu().numpy()
    
    
    for i,box in enumerate(boxes):
        x1 ,y1,x2,y2 ,conf,cls= box 
        x1 ,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        x1 = int(x1*orginal_width/640)
        y1 = int(y1*orginal_height/640)
        x2 = int(x2*orginal_width/640)
        y2 = int(y2*orginal_height/640)
        cropped_img = frame[y1:y2,x1:x2]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        label = f"{labels[int(cls)]} {conf:.2f}"
        cv2.putText(frame,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.imwrite(crop_output_dir+f"Cropped_imag_{last_detection}",cropped_img)
        frame_count = frame_count+1


    
    cv2.imshow('YOLOv5 Interface - Detect and Crop', frame)
    

# Define the directory to save frames
save_dir = "captured_frames"
os.makedirs(save_dir, exist_ok=True)

# Initialize a frame counter
frame_count = 0

if not cap.isOpened():
    print("Error: Unable to open the webcam")
else:
    while True:
        ret, frame = cap.read()
        


        if not ret:
            print("Failed to grab frame")
            break

        detect_and_draw_boxes(frame,save_dir,frame_count)
        
        
        
        
        
        #print(results.xyxy)
        
        
        # img_with_boxes = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
        
        
        # If any objects are detected, process and save the frame
        # if len(results[0].boxes) > 0:
        #     # Draw bounding boxes and labels on the frame
        #     annotated_frame = results[0].plot()  # This automatically draws bounding boxes and labels

        #     # Save the annotated frame as an image file
        #     frame_filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
        #     cv2.imwrite(frame_filename, annotated_frame)

        #     # Increment the frame counter
        #     frame_count += 1

        #     # Display the annotated frame with bounding boxes
        #     cv2.imshow('Webcam Video', annotated_frame)
        # else:
        #     # If no objects are detected, display the original frame
        #     cv2.imshow('Webcam Video', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()