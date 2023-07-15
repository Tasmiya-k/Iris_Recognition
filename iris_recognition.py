import cv2
import numpy as np
import os

# Path of the folder containing iris images
path = 'Iris Dataset'

# Read all the iris images in the folder
iris_images = {}
for file_name in os.listdir(path):
    name = os.path.splitext(file_name)[0]
    image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE)
    iris_images[name] = image

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect iris in the frame
    iris_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    iris = iris_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If an iris is detected, compare it with the known iris images
    if len(iris) > 0:
        (x, y, w, h) = iris[0]
        iris_roi = gray[y:y+h, x:x+w]
        iris_roi_resized = cv2.resize(iris_roi, (256, 64))
        iris_roi_normalized = cv2.normalize(iris_roi_resized, None, 0, 255, cv2.NORM_MINMAX)

        # Compare the iris with the known iris images
        match_found = False
        for name, iris_image in iris_images.items():
            iris_image_resized = cv2.resize(iris_image, (256, 64))
            iris_image_normalized = cv2.normalize(iris_image_resized, None, 0, 255, cv2.NORM_MINMAX)

            # Compare the two iris images using the correlation coefficient
            match_coeff = cv2.matchTemplate(iris_roi_normalized, iris_image_normalized, cv2.TM_CCORR_NORMED)[0][0]
            print('Match coefficient:', match_coeff)
            if match_coeff > 0.9:
                match_found = True
                break

        # Display the name of the matched iris image
        if match_found:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()