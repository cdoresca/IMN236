import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./Calibration_Images/test.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)

corners = cv.goodFeaturesToTrack(img_gray, maxCorners=300, qualityLevel=0.01, minDistance=50)
corners = np.int32(corners)

for corner in corners:
    x, y = corner.ravel()
    cv.circle(img, (x, y), 3, (0, 0, 255), -1)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Show the result
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
