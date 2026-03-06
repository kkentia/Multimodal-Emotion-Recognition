import cv2 

cap=cv2.VideoCapture(1) #default webcam 

if not cap.isOpened(): 
    print("could not open webcam")
    exit()

while True: 
    ret,frame = cap.read() #ret -> was frame captured successfully (True/False) frame -> image from webcam

    if not ret: 
        print("failed to grab frame")
        break

    cv2.imshow("Webcam Test", frame) #show the feed of the webcam 

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

cap.release()
cv2.destroyAllWindows()

