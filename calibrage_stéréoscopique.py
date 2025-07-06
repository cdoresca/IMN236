import numpy as np
import cv2 as cv
import Util as ut
import matplotlib.pyplot as plt



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


# Ouvrir image Droite et Gauche

imageDroite = cv.imread("babyD.ppm")
imageGauche = cv.imread("babyG.ppm")


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

imageDroite = cv.imread("AloeD.png")
imageGauche = cv.imread("AloeG.png")
'''

R = R_cam_g.T @ R_cam_d
T = R_cam_g.T @ (T_cam_d - T_cam_g)

e1 = T / np.linalg.norm(T)
e2 = np.array([ -T[1], T[0], 0]) / np.sqrt(T[0]**2 + T[1]**2)
e3 = np.cross(e1,e2)


Rg = np.array([e1,e2,e3])
Rd = R @ Rg


def rectification_inverse(img,O,S,z,R):
    
    height, width = img.shape[:2]
 
    m = np.zeros((height, width), dtype=np.float32)
    n = np.zeros((height, width), dtype=np.float32)

    for i in range(img.shape[1]):
        for j in range(img.shape[0]):

            x = (i-O[0]) * S[0]
            y = (j-O[1]) * S[1]

            q = np.array([x,y,z])
            
            Q = R.T @ q
            
            p = (z / Q[2]) * Q

            m[j,i] = (p[0] / S[0]) + O[0]
            n[j,i] = (p[1] / S[1]) + O[1]
                      
    img_rect = cv.remap(img, m, n, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    return img_rect





imgFinalD = rectification_inverse(imageDroite,O_cam_d,S_cam_d,zprime_cam_d,Rd)
imgFinalG = rectification_inverse(imageGauche,O_cam_g,S_cam_g,zprime_cam_g,Rg)

cv.imwrite('imgD.png',imgFinalD)
cv.imwrite('imG.png',imgFinalG)


plt.imshow(imgFinalG, cmap='gray')
plt.title('Image Gauche Rectifiée')
plt.axis('off')  
plt.show()


plt.imshow(imgFinalD, cmap='gray')
plt.title('Image Droite Rectifiée')
plt.axis('off')
plt.show()

