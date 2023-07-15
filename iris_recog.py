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

iris_codes = {}
for name, image in iris_images.items():
    # Apply Daugman's algorithm to extract the iris code
    rows, cols = image.shape
    radius = 40
    circle_x = cols // 2
    circle_y = rows // 2
    # Create a mask for the iris region
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (circle_x, circle_y), radius, 255, -1)
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    # Apply a Laplacian filter to enhance the iris region
    laplacian = cv2.Laplacian(masked_image, cv2.CV_64F)
    iris_code = np.zeros((64,), dtype=np.uint8)
    # Extract the iris code using the sign of the Laplacian values in the iris region
    for i in range(64):
        angle = 2 * np.pi * i / 64
        x = int(circle_x + radius * np.cos(angle))
        y = int(circle_y + radius * np.sin(angle))
        if x >= cols:
            x = cols - 1
        if y >= rows:
            y = rows - 1
        iris_code[i] = 1 if laplacian[y, x] >= 0 else 0
    iris_codes[name] = iris_code

print(iris_code)
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

    # If an iris is detected, perform iris recognition
    if len(iris) > 0:
        print("Iris is detected")
        
        (x, y, w, h) = iris[0]
        iris_roi = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Apply Daugman's algorithm to extract iris code
        circles = cv2.HoughCircles(iris_roi, cv2.HOUGH_GRADIENT, dp=2, minDist=iris_roi.shape[0]//8,
                                   param1=200, param2=100, minRadius=0, maxRadius=int(iris_roi.shape[0]/2))
        if circles is not None:
            print("Circle is not None")
            circles = np.round(circles[0, :]).astype("int")
            iris_code = ''
            for i, (cx, cy, r) in enumerate(circles):
                circle_roi = iris_roi[cy-r:cy+r, cx-r:cx+r]
                if circle_roi.shape[0] > 0 and circle_roi.shape[1] > 0:
                    circle_roi = cv2.resize(circle_roi, (32, 32))
                    mean = np.mean(circle_roi)
                    thresh = mean * 0.85
                    circle_roi[circle_roi <= thresh] = 0
                    circle_roi[circle_roi > thresh] = 1
                iris_code += ''.join(map(str, circle_roi.flatten().tolist()))
                print(iris_code)

            # Compare the iris code with the known iris codes using Hamming distance metric
            min_distance = 999999
            match_name = ''
            for name, iris_code_ref in iris_codes.items():
                print("I am here")
                distance = sum(c1 != c2 for c1, c2 in zip(iris_code, iris_code_ref))
                if distance < min_distance:
                    min_distance = distance
                    match_name = name
                    print(name)

            # Display the name of the matched iris image
            if min_distance < 100:
                print("min_distance<100")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, match_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()