import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def ncc(window1: np.ndarray, window2: np.ndarray) -> float:
    """
    Corrélation croisée centrée normalisée entre deux fenêtres.
    :param window1: Fenêtre extraite de l'image de gauche (numpy array 2D)
    :param window2: Fenêtre extraite de l'image de droite (numpy array 2D)
    :return: Coefficient de similarité (entre -1 et 1)
    """
    # Vérification des dimensions
    if window1.shape != window2.shape:
        raise ValueError("Les deux fenêtres doivent avoir les mêmes dimensions")

    # Conversion en float pour éviter les erreurs de type
    u = window1.astype(np.float32)
    v = window2.astype(np.float32)

    # Moyennes locales
    u_mean = np.mean(u)
    v_mean = np.mean(v)

    # Numérateur
    numerator = np.sum((u - u_mean) * (v - v_mean))

    # Dénominateur
    denominator = np.sqrt(np.sum((u - u_mean)**2) * np.sum((v - v_mean)**2))

    # Éviter la division par zéro
    if denominator == 0:
        return 0.0

    return numerator / denominator


def disparity_map_ncc(img_left: np.ndarray, img_right: np.ndarray, window_size: int, max_disparity: int) -> np.ndarray:
    """
    Calcule la carte de disparité entre deux images rectifiées en utilisant NCC.

    :param img_left: Image gauche (grayscale)
    :param img_right: Image droite (grayscale)
    :param window_size: demi-largeur de la fenêtre W → taille totale = 2W+1
    :param max_disparity: disparité maximale à tester (en pixels)
    :return: Carte de disparité (matrice 2D de même taille que l'image)
    """
    height, width = img_left.shape
    disparity = np.zeros((height, width), dtype=np.float32)
    W = window_size

    # Parcours de tous les pixels sauf les bords
    for y in tqdm(range(W, height - W), desc="Calcul de la carte de disparité"):
        for x in range(W + max_disparity, width - W):
            best_ncc = -1  # minimum possible pour NCC
            best_d = 0

            # Fenêtre dans l’image gauche
            window_left = img_left[y - W: y + W + 1, x - W: x + W + 1]

            # Recherche dans la ligne droite
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

# Calculer la carte de disparité
disp_map = disparity_map_ncc(img_left, img_right, window_size=5, max_disparity=64)

cv2.imwrite("output.png", disp_map.astype(np.uint8))

# Afficher la carte de disparité
plt.imshow(disp_map, cmap='gray')
plt.colorbar(label="Disparité (pixels)")
plt.show()