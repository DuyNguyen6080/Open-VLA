import cv2
import numpy as np
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open video capture device.")

while True:
    ret, frame = cap.read()

    if ret == True:

        

        cv2.imshow('frame',frame)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        continue

cap.release()
cv2.destroyAllWindows()