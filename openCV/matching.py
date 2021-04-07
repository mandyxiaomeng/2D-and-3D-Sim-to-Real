import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

def Matching (method, match, lowe_ratio):
    if method   == 'ORB':
        finder = cv.ORB_create()
    elif method == 'SIFT':
        finder = cv.SIFT_create()

    # BFMatcher with default params
    if match  == 'bf':
        matcher = cv.BFMatcher()

    kp1, des1 = finder.detectAndCompute(query_image,None)
    kp2, des2 = finder.detectAndCompute(train_image,None)

    matches = matcher.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good_matches= []

    for m,n in matches:
        if m.distance < lowe_ratio*n.distance:
            good_matches.append([m])
    

    return kp1, kp2, matches, good_matches

#read images
query_image = cv.imread('cam1.jpg',0)          # queryImage
train_image = cv.imread('cad1.png',0)          # trainImage


# rize images
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

#Matching images
method = 'ORB'  # 'SIFT'
match = 'bf'
lowe_ratio = 0.80

kp1, kp2, matches, good_matches = Matching (method, match, lowe_ratio)


#Result
msg1 = 'using %s with lowe_ratio %.2f' % (method, lowe_ratio)
msg2 = 'there are %d good matches' % (len(good_matches))

print(msg1)
print(msg2)

img3 = cv.drawMatchesKnn(query_image,kp1,train_image,kp2,good_matches, None, flags=2)

#print txt on the result image, save and plot result image
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img3,msg1,(10, 250), font, 0.8,(255,0,255),1,cv.LINE_AA)
cv.putText(img3,msg2,(10, 270), font, 0.8,(255,0,255),1,cv.LINE_AA)
fname = 'output3_%s_%.2f.jpg' % (method, lowe_ratio)
cv.imwrite(os.path.join('./output', fname), img3)

plt.imshow(img3),plt.show()