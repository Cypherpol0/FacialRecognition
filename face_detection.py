import cv2
import os

# VideoCapture object, with the argument for device index
cap = cv2.VideoCapture('Video_test2.mp4')  # opens video file for capturing frames
# Load pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# Create directory to save the collected data
data_dir = "collected_data_double"
os.makedirs(data_dir, exist_ok=True)

# Counter for labeling images
counter = 0

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret is False:
        continue

    # Detect objects (dictated by face_cascade) of different sizes from input,
    # returns as a list of rectangles, parameters are image, scalefactor, minNeighbors
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5) #frame name, scaling factor, number of neighbors

    if len(faces) == 0:
        continue

    # Process detected faces
    for i, face in enumerate(faces):
        x, y, w, h = face

        offset = 10

        # Ensure indices are within the valid range
        y_start = max(0, y - offset) #Calculates the starting y-coordinate for the region of interest (ROI) and ensures that it does not go below zero.
        y_end = min(frame.shape[0], y + h + offset) #Calculates the ending y-coordinate for the ROI and ensures that it does not exceed the height of the frame.
        x_start = max(0, x - offset) #Calculates the starting x-coordinate for the ROI and ensures that it does not go below zero.
        x_end = min(frame.shape[1], x + w + offset) #Calculates the ending x-coordinate for the ROI and ensures that it does not exceed the width of the frame.

        # Extract region of interest (ROI) around the detected face
        face_offset = frame[y_start:y_end, x_start:x_end]
        face_selection = cv2.resize(face_offset, (224, 224)) 

        # Save the face region to the data directory
        filename = os.path.join(data_dir, f"face_{counter}.png")
        cv2.imwrite(filename, face_selection)
        counter += 1
        # Display detected face and bounding box
        cv2.imshow(f"Face {i+1}", face_selection) # Create a separate window for each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display annotated frame with detected faces
    cv2.imshow("faces", frame)

    # Check for key press to quit capturing
    key_pressed = cv2.waitKey(1) & 0xff
    if key_pressed == ord('q'):
        break
# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()