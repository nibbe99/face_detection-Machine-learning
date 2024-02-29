import pandas as pd
import cv2


image = cv2.imread("humans.jpeg")

print(image)

face_cascade = cv2.CascadeClassifier("faces.xml")

faces = face_cascade.detectMultiScale(image, minNeighbors=4)    # neighbours.  Spaces between each face.

print(faces)    #GETS MATRICES, the kordinates. Draw rectable

for (a,b,c,d) in faces:
    cv2.rectangle(image, (a,b), (a+c, b+d), (255,255,255), 6)   #start and end of rectangle + colors + thick

cv2.imwrite("detect_faces.jpeg", image)