import cv2
cap = cv2.VideoCapture(0) #opens webcam and stores data in cap

while True: #if webcam is reading images, continue and false, break loop
    #capture frame by frame
    ret,frame = cap.read() #returns a bool ( True / False ). If the frame is read correctly, it will be True . 
    # if frame is read correctly ret is True

    if ret == False:
        continue

    cv2.imshow("video frame",frame) #Displays webcam reading in new window
    #frame = camera name
    
    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

cap.release() #close camera
cv2.destroyAllWindows()