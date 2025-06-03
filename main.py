import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from Util import Generate_Base_Points, Generate_Img_Points

#Generate_Base_Points()
img = cv.imread('test.jpg')
corners = Generate_Img_Points(img)

for corner in corners:
    x, y = corner.ravel()
    cv.circle(img, (x, y), 3, (0, 0, 255), -1)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')
plt.show()
