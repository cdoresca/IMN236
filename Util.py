import json
import cv2 as cv
import numpy as np
from collections import defaultdict


def Save_As_Json(array, file_name, path="./Points/"):

    data = {"points": array}

    with open(path+file_name, "w") as f:
        json.dump(data, f, indent=3)

def Open_Json(file_name, path="./Points/"):
    with open(path+file_name, "r") as f:
        return json.load(f)

def corners(img):
    corners = Generate_Img_Points(img)

    points = [tuple(pt.ravel()) for pt in corners]
    Remove_False_Corners(points, 650, 150)
    sorted = Sort_By_Columns(points)

    array = []
    index = 0
    for x, y in sorted:
        array.append({
            "point": index,
            "X": int(x),
            "Y": int(y),
            "Z": 0
        })
        index += 1
    Save_As_Json(array, "Test_Calibration_Grid.json")
    
def Generate_Base_Points(file_name = "Base_Points_Grid.json"):
    points = []
    point = 0

    for i in range(16):
        for j in range(16):
            points.append({
                "point": point,
                "X": i * 9,
                "Y": j * 9,
                "Z": 0
            })
            point += 1

    Save_As_Json(points, file_name)

def Generate_Img_Points(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)

    img_gray = cv.normalize(img_gray, None, 0, 255, cv.NORM_MINMAX)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)

    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    
    img_gray = cv.filter2D(img_gray, -1, sharpening_kernel)

    brightness = 10 
    contrast = 5
    img_gray = cv.addWeighted(img_gray, contrast, np.zeros(img_gray.shape, img_gray.dtype), 0, brightness)


    corners = cv.goodFeaturesToTrack(img_gray,
                                     maxCorners=260,
                                     qualityLevel=0.01,
                                     minDistance=2,
                                     mask=None,
                                     useHarrisDetector=True,
                                     k=0.04)

    corners = cv.cornerSubPix(np.float32(img_gray),
                              np.float32(corners).reshape(-1, 1, 2),
                              winSize=(5, 5), zeroZone=(-1, -1),
                              criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    
    corners = np.int32(corners)
    return corners

def Remove_False_Corners(points, xThreshold = 0, yThreshold = 0):
    index = 0
    for x,y in points:
        if x < xThreshold or y < yThreshold:
            points.pop(index) 
        index +=1

def Sort_By_Columns(points, tolerence = 15):
    columns = defaultdict(list)
    for x, y in points:
        group_x = round(x / tolerence) * tolerence
        columns[group_x].append((x, y))

    for group_x in columns:
        columns[group_x].sort(key=lambda p: p[1])

    sorted_by_columns = []
    for group_x in sorted(columns.keys()):
        sorted_by_columns.extend(columns[group_x])
    return sorted_by_columns