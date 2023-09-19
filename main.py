###################################
# !/usr/bin/python
# Python 3.10
# (C) 2023 admi.tech
###################################

# Import main packages
import cv2
import numpy as np

# Import additional files
# Define faces
face_classifier = cv2.CascadeClassifier('static/xml/haarcascade_frontalface_default.xml')

# Load Logo
logo = cv2.imread("static/img/logo2.png")
size = 100
logo = cv2.resize(logo, (size, size))
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

# Start Camera or Run Video File
# PLEASE SWITCH
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rawdata/Margot.mp4')
# cap = cv2.VideoCapture('rtsp://192.168.1.2:8080/out.h264')

# Functions

# Main Script
if __name__ == "__main__":

    # Main Loop
    while True:

        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        # Draw rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Logo Placement
        poi = img[-size-10:-10, -size-10:-10]
        poi[np.where(mask)] = 0
        poi += logo

        cv2.imshow(" Super Face Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # if frame is read correctly, ret is True
        if not ret:
            print("Can't retrieve frame - stream may have ended. Exiting..")
            break

    # Close program a close all windows
    cap.release()
    cv2.destroyAllWindows()
