
import os
import logging as log
import cv2
import sys

def timeLapse(startTime):
    return (cv2.getTickCount()-startTime)/cv2.getTickFrequency() 