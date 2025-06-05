import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import Util as ut

def corners(img):
    corners = ut.Generate_Img_Points(img)

    points = [tuple(pt.ravel()) for pt in corners]
    ut.Remove_False_Corners(points, 650, 150)
    sorted = ut.Sort_By_Columns(points)

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
    ut.Save_As_Json(array, "Test_Calibration_Grid.json")

img = cv.imread('test.jpg')

#Generate_Base_Points()
#corners(img)

z_prime = 26.0  # focal length in mm
Sx = 0.0014    # pixel size x in mm/pixel
Sy = 0.0014     # pixel size y in mm/pixel
Om = 2000/2       # principal point x in pixels
On = 1500/2     # principal point y in pixels

Mint = np.array([[z_prime / Sx, 0, Om],
                 [0, z_prime / Sy, On],
                 [0,  0,  1]])

R = np.eye(3)
T = np.zeros((3, 1))



data_original = ut.Open_Json("Base_Points_Grid.json")
data_calibration = ut.Open_Json("Test_Calibration_Grid.json")

points_original = data_original["points"]
points_calibration = data_calibration["points"]

pts_orig = np.array([[pt["X"], pt["Y"]] for pt in points_original], dtype=np.float32)
pts_calib = np.array([[pt["X"], pt["Y"]] for pt in points_calibration], dtype=np.float32)

x_dist=(pts_calib[:,0]-Om)*Sx
y_dist=(pts_calib[:,1]-On)*Sy

A=[]
b=[]

#index = np.random.randint(256,size=2) 
index=[0,5,48,63,112]

for i in index:
    A.append([y_dist[i]*pts_orig[i][0],y_dist[i]*pts_orig[i][1],y_dist[i],-x_dist[i]*pts_orig[i][0],-x_dist[i]*pts_orig[i][1]])
    b.append(x_dist[i])

A = np.array(A)
b = np.array(b)


P = np.linalg.pinv(A) @ b



S= P[0]**2+P[1]**2+P[3]**2+P[4]**2

Ty=0

if P[0]*P[4]-P[1]*P[3]!=0:
    Ty=(S-np.sqrt(S**2-4*(P[0]*P[4]-P[1]*P[3])**2))/(2*(P[0]*P[4]-P[1]*P[3])**2)
elif P[0]**2+P[3]**2!=0:
    Ty=1/(P[0]**2+P[3]**2)
    
elif P[1]**2+P[4]**2!=0:
    Ty=1/(P[1]**2+P[4]**2)
    
elif P[0]**2+P[1]**2!=0:
    Ty=1/(P[0]**2+P[1]**2)
elif P[3]**2+P[4]**2!=0:
    Ty=1/(P[3]**2+P[4]**2)


Ty=np.sqrt(Ty)

P=P*Ty
x_sign=P[0]*pts_orig[0][0]+P[1]*pts_orig[0][1]+P[2]
y_sign=P[3]*pts_orig[0][0]+P[4]*pts_orig[0][1]+Ty

if np.sign(x_sign)==np.sign(x_dist[0]) and  np.sign(y_sign)==np.sign(y_dist[0]):
    Ty=Ty
else:
    Ty*=-1
    P*=-1


r13=np.sqrt(1-P[0]**2-P[1]**2)


S=-1*np.sign(P[0]+P[1]+P[3]+P[4])

r23=S*np.sqrt(1-P[3]**2-P[4]**2)

L1=np.array([P[0],P[1],r13])
L2=np.array([P[3],P[4],r23])
L3=np.cross(L1,L2)


Tx=P[2]
R=np.array([L1,L2,L3])

A=[]
b=[]

#index = np.random.randint(256,size=2) 
index=[48,112]
for i in index:
    Y=R[1][0]*pts_orig[i][0]+R[1][1]*pts_orig[i][1]+Ty
    W=R[2][0]*pts_orig[i][0]+R[2][1]*pts_orig[i][1]

    A.append([Y, -y_dist[i]])
    b.append([W*y_dist[i]])

A=np.array(A)
b=np.array(b)
X=np.linalg.pinv(A) @ b

T = np.array([Tx,Ty])

r = x_dist**2 + y_dist**2
params = np.array([0,X[0].item(),X[1].item()])

denom = (R[2][0]*pts_orig[:, 0] + R[2][1]*pts_orig[:, 1] + params[2])



m = (params[1]/Sx)*((R[0][0]*pts_orig[:, 0] + R[0][1]*pts_orig[:, 1] + Tx) / denom)  +Om
n = (params[1]/Sy)*((R[1][0]*pts_orig[:, 0] + R[1][1]*pts_orig[:, 1] + Ty) / denom)  +On

plt.imshow(img)

plt.plot(m, n, 'ro')  # 'ro' = red + round marker
plt.title("Points sur l'image")
plt.axis('off')
plt.show()

params = ut.Levenberg_Marquardt(params,R,T,pts_orig,x_dist,r)

radiale = (1 + params[0] * r)
denom = (R[2][0]*pts_orig[:, 0] + R[2][1]*pts_orig[:, 1] + params[2])

m = (params[1]/Sx) * (((R[0][0]*pts_orig[:, 0] + R[0][1]*pts_orig[:, 1] + Tx) / denom) * radiale) + Om 
n = (params[1]/Sy) * (((R[1][0]*pts_orig[:, 0] + R[1][1]*pts_orig[:, 1] + Ty) / denom) * radiale) + On 

plt.imshow(img)

error = np.sqrt(np.mean((m - pts_calib[:, 0])**2 + (n - pts_calib[:, 1])**2))
print(f"Erreur de reprojection RMS : {error:.2f} pixels")

plt.plot(m, n, 'ro')  # 'ro' = red + round marker
plt.title("Points sur l'image")
plt.axis('off')
plt.show()







#================================================================================================================================
#for idx, (x, y) in enumerate(sorted):
#    cv.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
#    cv.putText(img, str(idx), (int(x) + 5, int(y) - 5), cv.FONT_HERSHEY_SIMPLEX, 
#               0.5, (0, 255, 0), 1, cv.LINE_AA)
#    
#
#plt.imshow(img)
#plt.axis('off')
#plt.show()