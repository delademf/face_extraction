import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd

img = cv.imread('imgees/i.jpg')

plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(static_image_mode=True)

results = face_mesh.process(cv.cvtColor(img,cv.COLOR_BGR2RGB))
# print(results.multi_face_landmarks)
landmarks = results.multi_face_landmarks[0]

face_oval = mp_face.FACEMESH_FACE_OVAL

df = pd.DataFrame(list(face_oval),columns= ['p1','p2'])
print (df.head)
print (df.shape)

routes_idx = []

pl = df.iloc[0]['p1']
p2 = df.iloc[0]['p2']

for i in range (0,df.shape[0]):
    obj = df[df['p1']==p2]
    p1 = obj['p1'].values[0]
    p2 = obj['p2'].values[0]

    current_route = [ ]
    current_route.append(p1)
    current_route.append(p2)
    routes_idx.append(current_route)

for route_idx in routes_idx:
    print(f"Draw a line from ({route_idx[0]}) landmark point to ({route_idx[1]}) landmark point")


routes =[]

for source_idx,target_idx in routes_idx:
    source = landmarks.landmark[source_idx]
    target = landmarks.landmark[target_idx]
    

    relative_source = int(source.x * img.shape[1]),int(source.y * img.shape[0])
    relative_target = int(target.x * img.shape[1]),int(target.y * img.shape[0])


    routes.append(relative_source)
    routes.append(relative_target)




mask = np.zeros((img.shape[0],img.shape[1]))
mask = cv.fillConvexPoly(mask,np.array(routes),1)
mask = mask.astype(bool)

cut = np.zeros_like(img)
cut[mask] = img[mask]


plt.imshow(cv.cvtColor(cut,cv.COLOR_BGR2RGB))

plt.show()

