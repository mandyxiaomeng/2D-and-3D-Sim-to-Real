import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


query_image = cv.imread('cam1.jpg',0)          # queryImage
train_image = cv.imread('cad1.png',0)          # trainImage

#print(query_image.shape)
#print(train_image.shape)
#print(query_image.size)
#print(train_image.size)

scale_percent = ((train_image.shape[1])/(query_image.shape[1]))/(50/100)

width = int(query_image.shape[1] * scale_percent)
height = int(query_image.shape[0] * scale_percent)

query_image = cv.resize(query_image,(width,height))

#print(query_image.shape)
#print(query_image.size)

method = 'ORB'  # 'SIFT'
lowe_ratio = 0.80

if method   == 'ORB':
    finder = cv.ORB_create()
elif method == 'SIFT':
    finder = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = finder.detectAndCompute(query_image,None)
kp2, des2 = finder.detectAndCompute(train_image,None)

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

img3 = cv.drawMatchesKnn(query_image,kp1,train_image,kp2,good, None, flags=2)

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img3,msg1,(10, 250), font, 0.8,(255,0,255),1,cv.LINE_AA)
cv.putText(img3,msg2,(10, 270), font, 0.8,(255,0,255),1,cv.LINE_AA)
fname = 'output3_%s_%.2f.jpg' % (method, lowe_ratio)
#cv.imwrite(fname, img3)
cv.imwrite(os.path.join('./output', fname), img3)

plt.imshow(img3),plt.show()