import cv2 as cv
import matplotlib.pyplot as plt
from Util import Generate_Base_Points, Generate_Img_Points, Remove_False_Corners, Sort_By_Columns, Save_As_Json

#Generate_Base_Points()
img = cv.imread('test.jpg')
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

for idx, (x, y) in enumerate(sorted):
    cv.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv.putText(img, str(idx), (int(x) + 5, int(y) - 5), cv.FONT_HERSHEY_SIMPLEX, 
               0.5, (0, 255, 0), 1, cv.LINE_AA)
    

plt.imshow(img)
plt.axis('off')
plt.show()
