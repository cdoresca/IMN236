import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import cv2.ximgproc as xip


def ncc(window1: np.ndarray, window2: np.ndarray) -> float:

    u = window1.astype(np.float32)
    v = window2.astype(np.float32)

    u_mean = np.mean(u)
    v_mean = np.mean(v)

    numerator = np.sum((u - u_mean) * (v - v_mean))

    denominator = np.sqrt(np.sum((u - u_mean)**2) * np.sum((v - v_mean)**2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def disparity_map_ncc(img_left: np.ndarray, img_right: np.ndarray, window_size: int, max_disparity: int) -> np.ndarray:
    height, width = img_left.shape
    disparity = np.zeros((height, width), dtype=np.float32)
    W = window_size

    for y in tqdm(range(W, height - W), desc="Calcul de la carte de disparité"):
        for x in range(W + max_disparity, width - W):
            best_ncc = -1
            best_d = 0

            window_left = img_left[y - W: y + W + 1, x - W: x + W + 1]

            for d in range(max_disparity + 1):
                x_right = x - d
                if x_right - W < 0:
                    continue
                window_right = img_right[y - W: y + W + 1, x_right - W: x_right + W + 1]
                sim = ncc(window_left, window_right)
                if sim > best_ncc:
                    best_ncc = sim
                    best_d = d

            disparity[y, x] = best_d

    return disparity

img_left = cv2.imread("images/im0.png", cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread("images/im1.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.float32)/25

disp_map = disparity_map_ncc(img_left, img_right, window_size=5, max_disparity=64)

disp_map_filtered = cv2.medianBlur(disp_map.astype(np.uint8), 5)
disp_map_filtered = cv2.bilateralFilter(disp_map_filtered, 9, 75, 75)

cv2.imwrite("output.png", disp_map_filtered.astype(np.uint8))

plt.imshow(disp_map, cmap='gray')
plt.colorbar(label="Disparité (pixels)")
plt.show()