import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


def ncc(fenetre1: np.ndarray, fenetre2: np.ndarray) -> float:

    u = fenetre1.astype(np.float32)
    v = fenetre2.astype(np.float32)

    u_mean = np.mean(u)
    v_mean = np.mean(v)

    numerateur = np.sum((u - u_mean) * (v - v_mean))

    denominateur = np.sqrt(np.sum((u - u_mean)**2) * np.sum((v - v_mean)**2))

    if denominateur == 0:
        return 0.0

    return numerateur / denominateur


def disparity_map_ncc(img_left: np.ndarray, img_right: np.ndarray, window_size: int, max_disparity: int) -> np.ndarray:
    height, width = img_left.shape
    disparity = np.zeros((height, width), dtype=np.float32)
    W = window_size

    for y in tqdm(range(W, height - W), desc="Calcul de la carte de disparité"):
        deja_match = np.zeros(width, dtype=bool) #Initialisation unicite

        for x in range(W + max_disparity, width - W):
            best_ncc = 0
            best_d = 0
            best_x_right = 0

            fenetre_left = img_left[y - W: y + W + 1, x - W: x + W + 1]

            for d in range(max_disparity + 1):
                x_right = x - d

                if x_right - W < 0:
                    continue
                if deja_match[x_right]: #Si deja match on ignore
                    continue

                fenetre_right = img_right[y - W: y + W + 1, x_right - W: x_right + W + 1]
                sim = ncc(fenetre_left, fenetre_right)

                if sim > best_ncc:
                    best_ncc = sim
                    best_d = d
                    best_x_right = best_d

            #Gestion de l'ordre
            if best_x_right >= 0:
                if x > W:
                    if disparity[y, x - 1] + W< best_d:
                        best_d = disparity[y, x - 1] + W

                deja_match[best_x_right] = True  # Marquer que le pixel a un match

            disparity[y, x] = best_d

    return disparity


def depth_map(disparity_map: np.ndarray, z_prime: float, dOx: float, Tx: float) -> np.ndarray:
    height, width = disparity_map.shape
    Zc = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):

            denominator = disparity_map[y, x] + dOx

            if denominator == 0:
                Zc[y, x] = 0
            else:
                Zc[y, x] = (z_prime * Tx) / denominator
    
    return Zc

img_left = cv2.imread("images/im0.png", cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread("images/im1.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.float32)/25

disp_map = disparity_map_ncc(img_left, img_right, 5, 60)
dep_map = depth_map(disp_map, 1500, 52, 80)

cv2.imwrite("disp.png", disp_map.astype(np.uint8))
cv2.imwrite("depth.png", dep_map.astype(np.uint8))

plt.imshow(disp_map, cmap='gray')
plt.colorbar(label="Disparité (pixels)")
plt.show()

plt.imshow(dep_map, cmap='gray')
plt.colorbar(label="Profondeur (pixels)")
plt.show()