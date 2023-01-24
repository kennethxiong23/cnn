import cv2 as cv
from time import sleep

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("camercan failed to open quit")
    quit()

else:
    print("camera reday for pcitures")

sleep(1)

for i in range(100):
    _, frame = cap.read()
    cv.imwrite(f'images/Kenneth/img_{i}.jpg', frame)

input("leav")

for i in range(100):
    _, frame = cap.read()
    cv.imwrite(f'images/noFace/img_{i}.jpg', frame)