import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from Util import Generate_Base_Points, Generate_Img_Points, Remove_False_Corners, Sort_By_Columns, Save_As_Json, Open_Json

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


img = cv.imread('test.jpg')
#Generate_Base_Points()
#corners(img)

z_prime = 26.0  # focal length in mm
Sx = 0.0008    # pixel size x in mm/pixel
Sy = 0.0008     # pixel size y in mm/pixel
Om = 1000       # principal point x in pixels
On = 750       # principal point y in pixels

Mint = np.array([[z_prime / Sx, 0, Om],
                 [0, z_prime / Sy, On],
                 [0,  0,  1]])

R = np.eye(3)
T = np.zeros((3, 1))


data_original = Open_Json("Base_Points_Grid.json")
data_calibration = Open_Json("Test_Calibration_Grid.json")

points_original = data_original["points"]
points_calibration = data_calibration["points"]

pts_orig = np.array([[pt["X"], pt["Y"]] for pt in points_original], dtype=np.float32)
pts_calib = np.array([[pt["X"], pt["Y"]] for pt in points_calibration], dtype=np.float32)














#M_affine = cv.estimateAffine2D(pts_orig, pts_calib)[0]
#pts_transformed_affine = cv.transform(pts_orig[None, :, :], M_affine)[0]


#plt.figure(figsize=(10, 8))
#plt.scatter(pts_calib[:, 0], pts_calib[:, 1], label='Points calibrés (référence)', c='red')
#plt.scatter(pts_transformed_affine[:, 0], pts_transformed_affine[:, 1], label='Transformés (affine)', c='blue')
#
#plt.legend()
#plt.show()
#=================================================================================================================================
#for idx, (x, y) in enumerate(sorted):
#    cv.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
#    cv.putText(img, str(idx), (int(x) + 5, int(y) - 5), cv.FONT_HERSHEY_SIMPLEX, 
#               0.5, (0, 255, 0), 1, cv.LINE_AA)
#    
#
#plt.imshow(img)
#plt.axis('off')
#plt.show()