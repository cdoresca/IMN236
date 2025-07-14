import numpy as np
import cv2 as cv
import Util as ut
import matplotlib.pyplot as plt


def rectification_inverse(img, O, S, z, R):
    img_rect = np.zeros_like(img)


    for i in range(img.shape[1]):
        for j in range(img.shape[0]):

            x = (i-O[0]) * S[0]
            y = (j-O[1]) * S[1]

            q = np.array([x,y,z])
            Q = R.T @ q
            p = (z / Q[2]) * Q

            m = (p[0] / S[0]) + O[0]
            n = (p[1] / S[1]) + O[1]
            
            
            img_rect[j,i] = ut.bilinear_interpolation(img,m,n)

    return img_rect

'''
# Paramètres de la caméra de gauche
R_cam_g = np.array([[ 0.9962, -0.0015, -0.0871],
                    [0, 0.996, -0.0174],
                    [ 0.0872, 0.0174, 0.996]])
T_cam_g = np.array([3, 5, 7])
O_cam_g = np.array([512.843, 493.819])
S_cam_g = np.array([0.00155227, 0.00155227])
zprime_cam_g = 1.0

# Paramètres de la caméra de droite
R_cam_d = np.array([[ 0.9962, -0.0015, 0.0871 ],
                    [ 0, 0.996, 0.0174],
                    [ -0.0872, -0.0174, 0.996]])
T_cam_d = np.array([5.6, 5, 7])
O_cam_d = np.array([726.753, 530.523])
S_cam_d = np.array([0.00155227, 0.00155227])
zprime_cam_d = 1.0

imageDroite = cv.imread("images/babyD.ppm")
imageGauche = cv.imread("images/babyG.ppm")
'''

# Paramètres de la caméra de gauche
R_cam_g = np.array([[ 0.9962, 0, -0.0872],
                    [0, 0.9962, 0],
                    [ 0.0872, 0, 0.9962]])
T_cam_g = np.array([0, 0, 0])
O_cam_g = np.array([538.625, 510.471])
S_cam_g = np.array([0.00155227, 0.00155227])
zprime_cam_g = 1.0

# Paramètres de la caméra de droite
R_cam_d = np.array([[ 0.9962, 0, 0.0872 ],
                    [ 0, 0.9962, 0],
                    [ -0.0872, 0, 0.9962]])

T_cam_d = np.array([5, 0, 0])
O_cam_d = np.array([765.134, 510.599])
S_cam_d = np.array([0.00155227, 0.00155227])
zprime_cam_d = 1.0

imageDroite = cv.imread("images/AloeD.png")
imageGauche = cv.imread("images/AloeG.png")


R = R_cam_g.T @ R_cam_d
T = R_cam_g.T @ (T_cam_d - T_cam_g)

e1 = T / np.linalg.norm(T)
e2 = np.array([ -T[1], T[0], 0]) / np.linalg.norm([-T[1],T[0]])
e3 = np.cross(e1,e2)

Rg = np.array([e1,e2,e3])
Rd = R @ Rg

threads = []

imgG = ut.CustomThread(target=rectification_inverse,args=(imageGauche,O_cam_g,S_cam_g,zprime_cam_g,Rg))
imgD = ut.CustomThread(target=rectification_inverse,args=(imageDroite,O_cam_d,S_cam_d,zprime_cam_d,Rd))

threads.extend([imgG,imgD])

for t in threads:
    t.start()

imgFinalD =imgD.join() 
imgFinalG = imgG.join()

cv.imwrite('images/imgD.png',imgFinalD)
cv.imwrite('images/imG.png',imgFinalG)