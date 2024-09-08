import cv2
import numpy as np
import tempfile
from PIL import Image as img
import os
import requests
from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np


def check_readability(ret):
    if not ret: 
        print("Could not read frame")
        exit()
    else: 
        print("Frame is read")


#Capture Video
video_path = "C:/Users/JP/Downloads/12267384_3840_2160_30fps.mp4"
capture = cv2.VideoCapture(video_path)

#Checks if video opens 
if not capture.isOpened():
    print("Video cannot be openned")
    exit()

#Set Custom Frame
num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
ret, frame = capture.read()
check_readability(ret)


#Use Sam model to mask images 
model_path = "sam_vit_h_4b8939.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = sam_model_registry["vit_h"](pretrained = False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

predictor = SamPredictor(model)
predictor.initialize_tracking(frame)

while True: 
    ret, frame = capture.read()
    if not ret: 
        break

    masks,scores = predictor.track(frame)

    for mask, score in zip(mask,scores):
        print(f"Mask shape: {mask.shape}, Score:{score}")

capture.release()

'''
#Train YOLO through each frame 
while capture.isOpened():
    
    ret, frame = capture.read()

    check_readability(ret)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)


    #Press q to terminate
    print("Press q to terminate manually") 
    if cv2.waitKey(1) == ord('q'):
        break
        
    
    #Remove temp image
    os.remove(temp_path) 
'''
