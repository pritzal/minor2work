import cv2
import numpy as np

# Define the lower and upper boundaries for the colors you want to detect
lower_bound = np.array([0, 100, 100])
upper_bound = np.array([20, 255, 255])

# Create a video capture object for the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Convert the frame to HSV format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask using the lower and upper bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask to the frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Perform morphological operations to close gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)

        # Calculate the circularity of the contour
        circularity = 4 * 3.14 * area / (perimeter ** 2)

        # If the contour is large enough and circular enough
        if area > 100 and circularity > 0.8:
            # Print the size information of the contour
            print('Area:', area)
            print('Perimeter:', perimeter)
            print('Circularity:', circularity)

            # Draw a bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get the color of the object
            color = frame[y + h // 2, x + w // 2]
            print('Color:', color)

    # Show the frame with the contour drawn
    cv2.imshow('Object Detection', frame)

    # Check for a key press and exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
