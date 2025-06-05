import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial.transform import Rotation as R_scipy
from Util import Open_Json

# === PARAMÈTRES CAMÉRA ===
Zprime = 4       # Focale en mm
Sx = Sy = 0.0014 # Taille pixel en mm
Om, On = 1250, 750  # Centre image

# === MATRICE INTRINSÈQUE ===
Mint = np.array([
    [Zprime / Sx, 0, Om],
    [0, Zprime / Sy, On],
    [0, 0, 1]
])

# === CHARGEMENT DES POINTS ===
data_original = Open_Json("Base_Points_Grid.json")
data_calibration = Open_Json("Test_Calibration_Grid.json")

points_objet = np.array([[pt["X"], pt["Y"], pt["Z"]] for pt in data_original["points"]], dtype=np.int64)
points_monde = np.array([[pt["X"], pt["Y"]] for pt in data_calibration["points"]], dtype=np.int64)

# === TRANSFORMATION EN COORDONNÉES EN MM ===
Xdi = (points_monde[:, 0] - Om) * Sx
Ydi = (points_monde[:, 1] - On) * Sy

# === ÉTAPE 0 : Résolution des coefficients ===
indices = [0, 5, 48, 63, 112]
A = []
b = []

for i in indices:
    X, Y = points_objet[i][:2]
    xdi, ydi = Xdi[i], Ydi[i]
    A.append([ydi * X, ydi * Y, ydi, -xdi * X, -xdi * Y])
    b.append([xdi])

A = np.array(A)
b = np.array(b)
x = np.linalg.solve(A, b)

r11, r12, Tx, r21, r22 = x.flatten()

# === ÉTAPE 2 : Calcul de Ty ===
Sr = r11**2 + r12**2 + r21**2 + r22**2
det = r11 * r22 - r12 * r21

if det != 0:
    Ty = (Sr - np.sqrt(Sr**2 - 4 * det**2)) / (2 * det)
else:
    Ty = 1 / (r11**2 + r21**2 or r12**2 + r22**2)

Ty = abs(Ty)  # Forcer signe positif, inversé plus tard si nécessaire

# === ÉTAPE 3 : Matrice de rotation + Tx ===
R11, R12 = r11 * Ty, r12 * Ty
R21, R22 = r21 * Ty, r22 * Ty
Tx *= Ty

# Vérification du signe de Ty
Xs, Ys, _ = points_objet[0]
x_tild = R11 * Xs + R12 * Ys + Tx
y_tild = R21 * Xs + R22 * Ys + Ty

if not (np.sign(x_tild) == np.sign(Xdi[0]) and np.sign(y_tild) == np.sign(Ydi[0])):
    Ty = -Ty
    R11, R12, R21, R22, Tx = -R11, -R12, -R21, -R22, -Tx

# Matrice de rotation 3x3
L1 = np.array([R11, R12, np.sqrt(1 - R11**2 - R12**2)])
L2 = np.array([R21, R22, np.sqrt(1 - R21**2 - R22**2)])
L3 = np.cross(L1, L2)
R = np.vstack([L1, L2, L3])

# === ÉTAPE 5 : Estimation Zprime et Tz ===
A = []
b = []

for i in [48, 112]:
    X, Y = points_objet[i][:2]
    Y_val = R[1, 0]*X + R[1, 1]*Y + Ty
    W_val = R[2, 0]*X + R[2, 1]*Y

    A.append([Y_val, -Ydi[i]])
    b.append([W_val * Ydi[i]])

x = np.linalg.solve(np.array(A), np.array(b))
Tz, Zprime_est = x.flatten()

T = np.array([[Tx], [Ty], [Tz]])


def Calculer_erreur(params):
    rvec = params[0:3]
    tvec = params[3:6]
    Zprime = params[6]
    k1 = params[7]

    R_mat = R_scipy.from_rotvec(rvec).as_matrix()

    erreurs = []
    for i in range(len(points_objet)):
        X, Y, Z = points_objet[i]
        pt3D = np.array([X, Y, Z])

        X_cam = R_mat @ pt3D + tvec
        x = X_cam[0]
        y = X_cam[1]
        w = X_cam[2]

        xd = Zprime * x / w
        yd = Zprime * y / w
        r2 = xd**2 + yd**2
        xd_corr = xd * (1 + k1 * r2)
        yd_corr = yd * (1 + k1 * r2)

        u = xd_corr / Sx + Om
        v = yd_corr / Sy + On

        u_meas, v_meas = points_monde[i]
        erreurs.extend([u - u_meas, v - v_meas])

    return np.array(erreurs) 

def Calcule_Jacobien(params):
    n_params = len(params)

    e0 = Calculer_erreur(params)
    J = np.zeros((len(e0), n_params))

    for j in range(n_params):
        delta = np.zeros_like(params)
        delta[j] = 0.000001
        el = Calculer_erreur(params + delta)
        J[:, j] = (el -e0)/0.000001
    return J


def Levenber_Marquardt(params, kmax):
    k = 0
    v = 2
    p = params

    J = Calcule_Jacobien(params)
    e = Calculer_erreur(params)
    Jprime = np.transpose(J)

    A = Jprime * J
    g = Jprime * e

    trouve = max(np.linalg.norm(g)) <= sys.float_info.epsilon
    τ = 2 * np.pi
    μ = τ * max(np.diag(A))

    while not trouve or k< kmax:
        k = k+1
        h = (A + μ * np.identity(n))**-1*(-g)
        if(np.linalg.norm(h) <= sys.float_info.epsilon * np.linalg.norm(p)):
            trouve = True
        else:
            Pnew = p + h
            pp = (Calculer_erreur(p) - Calculer_erreur(Pnew)) / (np.transpose(h) * (μ * h - g))

            if pp > 0:
                p = Pnew
                J = Calcule_Jacobien(p)
                Jprime = np.transpose(J)
                e = Calculer_erreur(p)

                A = Jprime*J
                g = Jprime*e
            
                trouve = max(np.linalg.norm(g)) <= sys.float_info.epsilon

                μ = μ * max(1/3, 1 - (2*pp - 1)**3)
                v = 2
            else:
                μ = v * μ
                v = 2 * v
                

























# === ÉTAPE 6 : Correction radiale (k1) ===
xd_list = []
yd_list = []
r2_list = []
dx_list = []
dy_list = []

A = []
b = []

for i in range(len(points_objet)):
    X, Y, Z = points_objet[i]

    xd = Zprime_est * (R[0] @ [X, Y, Z] + T[0]) / (R[2] @ [X, Y, Z] + T[2])
    yd = Zprime_est * (R[1] @ [X, Y, Z] + T[1]) / (R[2] @ [X, Y, Z] + T[2])
    r2 = xd**2 + yd**2

    xd_list.append(xd)
    yd_list.append(yd)
    r2_list.append(r2)
    dx = Xdi[i] - xd
    dy = Ydi[i] - yd

    A.append([xd * r2])
    A.append([yd * r2])
    b.append([dx])
    b.append([dy])

A = np.array(A)
b = np.array(b)

k1 = np.linalg.solve(A, b)[0, 0]

# === ÉTAPE 7 : Reprojection des points corrigés ===
proj_pts = []

for i in range(len(points_objet)):

    xd_corr = xd_list[i] * (1 + k1 * r2_list[i])
    yd_corr = yd_list[i] * (1 + k1 * r2_list[i])

    m = xd_corr / Sx + Om
    n = yd_corr / Sy + On

    proj_pts.append([m, n])

proj_pts = np.array(proj_pts)

# === AFFICHAGE DES POINTS REPROJETÉS ===
img = cv.imread('test.jpg')
plt.figure(figsize=(12, 9))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c='red', marker='x', s=20, label='Reprojetés')
plt.legend()
plt.show()
