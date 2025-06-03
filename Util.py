import json
import cv2 as cv
import numpy as np


def Generate_Base_Points():
    points = []
    point = 1

    for i in range(16):
        for j in range(16):
            points.append({
                "point": point,
                "X": i * 9,
                "Y": j * 9,
                "Z": 0
            })
            point += 1

    data = {"points": points}

    with open("../Points/Base_Point_grid.json", "w") as f:
        json.dump(data, f, indent=3)

def Generate_Img_Points(img):
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)

    brightness = 10 
    contrast = 2.3  

    img_gray = cv.addWeighted(img_gray, contrast, np.zeros(img_gray.shape, img_gray.dtype), 0, brightness)
    
    corners = cv.goodFeaturesToTrack(img_gray, maxCorners=250, qualityLevel=0.01, minDistance=10)
    corners = np.int32(corners)

    return corners

