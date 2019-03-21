"""
import cv2
import numpy

import os

video_path  = "***.mp4"
cap = cv2.VideoCapture(video_path)
ret = True
while(ret):
    ret, frame = cap.read()
    img = frame
    # cv2.imshow('frame_img',img)
    cv2.imwrite("frame_image.png", frame)
cap.release()

"""

