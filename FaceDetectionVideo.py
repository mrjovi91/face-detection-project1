import cv2

# Load detection algorithm for frontal face detection
face_detection_dataset = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load webcam video into OpenCV
webcam = cv2.VideoCapture(0)
if webcam.isOpened():
    while True:
        # Read current Frame
        successful_frame_read, frame = webcam.read()

        # Convert frame to black and white for face detection
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces from grayscale image and get coordinates
        face_coordinates = face_detection_dataset.detectMultiScale(grayscale)

        # Draw squares around faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Show image
        cv2.imshow('Test', frame)

        # Exit on key press
        key = cv2.waitKey(1)

        # Stop if Q key is pressed
        if key==81 or key==113:
            break
    webcam.release()



print("Done")