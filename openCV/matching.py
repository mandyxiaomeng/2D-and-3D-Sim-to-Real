import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


img1 = cv.imread('cam1.jpg',0)          # queryImage
img2 = cv.imread('cad1.png',0) # trainImage

print(img1.shape)
img1 = cv.resize(img1,(800,600))

method = 'ORB'  # 'SIFT'
lowe_ratio = 0.80

if method   == 'ORB':
    finder = cv.ORB_create()
elif method == 'SIFT':
    finder = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = finder.detectAndCompute(img1,None)
kp2, des2 = finder.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []

for m,n in matches:
    if m.distance < lowe_ratio*n.distance:
        good.append([m])

msg1 = 'using %s with lowe_ratio %.2f' % (method, lowe_ratio)
msg2 = 'there are %d good matches' % (len(good))

print(msg1)
print(msg2)

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)

#font = cv.FONT_HERSHEY_SIMPLEX
#cv.putText(img3,msg1,(10, 250), font, 0.8,(255,0,255),1,cv.LINE_AA)
#cv.putText(img3,msg2,(10, 270), font, 0.8,(255,0,255),1,cv.LINE_AA)
fname = 'output3_%s_%.2f.jpg' % (method, lowe_ratio)
#cv.imwrite(fname, img3)
cv.imwrite(os.path.join('./output', fname), img3)

plt.imshow(img3),plt.show()