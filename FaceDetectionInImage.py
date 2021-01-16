import cv2

# Load detection algorithm for frontal face detection
face_detection_dataset = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image into OpenCV
original = cv2.imread('img\\test4.jpg')

# Convert to black and white for face detection
grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Detect faces from grayscale image and get coordinates
face_coordinates = face_detection_dataset.detectMultiScale(grayscale)

# Draw squares around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(original, (x,y), (x+w, y+h), (0,255,0), 2)

# Show image
cv2.imshow('Test', original)

# Exit on key press
cv2.waitKey()

print("Done")