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

def calculer_erreurs(params,R,T,pts_monde,xd,r):
    e = []
    for i, (x,y) in enumerate(pts_monde):
        e.append((xd[i] * (1 + params[0] * r[i]) * (R[2][0] * x + R[2][1] * y + params[2]) - (R[0][0] * x + R[0][1] * y + T[0])) * params[1])
    return np.array(e)   

def jacobien(params,R,T,pts_monde,xd,r):

    n_params = len(params)

    e0 = calculer_erreurs(params,R,T,pts_monde,xd, r)

    J = np. zeros((len(e0),n_params))

    for j in range(n_params):
        delta = np.zeros_like(params)
        delta[j]=0.000001
        e1 = calculer_erreurs(params + delta,R,T,pts_monde,xd,r)
        J[:,j] = (e1 - e0) / 0.000001
    
    return J

def Levenberg_Marquardt(p0,R,T,pts_monde,xd,r):

    k = 0
    v = 2
    
    p = p0.copy()

    J = jacobien(p,R,T,pts_monde,xd,r)
    e = calculer_erreurs(p,R,T,pts_monde,xd,r)
    A = J.T @ J
    g = J.T @ e

    find = max(np.abs(g)) <= 1e-6
    
    u= 1e-3 * np.max(A.diagonal())

    while not find:
        k += 1
        h = np.linalg.inv(A + u * np.eye(len(p))) @ (-1 * g)
        print(h)
        if np.linalg.norm(h) <= 1e-6 * np.linalg.norm(p):
            find = True 

        else:
            p_new = p + h
            e_new = calculer_erreurs(p_new,R,T,pts_monde,xd,r)
            rho = float((np.linalg.norm(e)**2 - np.linalg.norm(e_new)**2) / (h.T @ (u * h - g)))

            if rho > 0:
                p=p_new
                J = jacobien(p,R,T,pts_monde,xd,r)
                e = calculer_erreurs(p,R,T,pts_monde,xd,r)
                A = J.T @ J
                g = J.T @ e
                find = max(np.abs(g)) <= 1e-6
                u *= max(1/3, 1 - (2 * rho - 1)**3)
                v=2

            else:

                u *= v
                v *= 2
    
    return p
