import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

imgs = [cv.imread(file) for file in sorted(glob.glob("./images/binary/*.ppm"))]
grays = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imgs]

res = np.zeros(grays[0].shape, dtype=np.uint16)

for i in range(0, len(grays), 2):
    img = grays[i]
    inv = grays[i + 1]

    bit_index = i // 2
    bit_mask = 1 << bit_index

    res[img < inv] += bit_mask


ET = np.std(res)
mean = np.mean(res)

res[res < mean -2*ET] = mean -2*ET
res[res > mean +2*ET] = mean +2*ET

res = cv.normalize(res, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

plt.imshow(res, cmap='gray')
plt.colorbar()
plt.show()

cv.imwrite("./images/res.png", res.astype(np.uint8))
