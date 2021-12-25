import numpy as np
from preprocessing import *
from math import sin,cos,pi
import cv2
import matplotlib.pyplot as plt

def Diacritics_Segmentation(binarized):
    hist = np.sum(1-binarized/255,axis=1)
    baseline = np.argmax(hist)

    im_floodfill = binarized.copy()

    hImg, wImg = binarized.shape
    mask = np.zeros((hImg+2, wImg+2), np.uint8)
    for col in range(1,binarized.shape[1]):
        if binarized[baseline,col-1]==0 and binarized[baseline,col]==255:
            #flood fill
            cv2.floodFill(im_floodfill, mask, (col-1, baseline), 255)
    
    cv2.imwrite("floodfill.png",im_floodfill)

    contours,_ = cv2.findContours(255-im_floodfill, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    # demo= np.zeros((hImg, wImg), np.uint8)
    # cv2.drawContours(demo, contours, -1, 255, -1)
    # cv2.imwrite("contours.png",demo)
    shapes=[]
    for i,contour in enumerate(contours):
        x,y,w,h  = cv2.boundingRect(contour)
        if w<=3 or h<=3:
            continue
        contourImg =255- binarized[y:y+h,x:x+w].copy()
        shapes.append(contourImg)
        cv2.imwrite(f"diacritics/temp_{i}.png",contourImg)
    return shapes

def extractFeatures_CPRP(shape):
    CP=[]
    PR=[]
    whitePixels = np.array(np.where(shape == 255))
    whitePixels = whitePixels.T
    mid = len(whitePixels)//2
   
   
    #centroid point
    x0,y0 = whitePixels[mid][1], whitePixels[mid][0]
    # print(x0,y0)
    # print(whitePixels)

    R = max([ np.sqrt((pixel[0]-y0)**2 + (pixel[1]-x0)**2) for pixel in whitePixels])
    R = int(np.floor(R))


    for gama in range(0,R):
        Fgama =0
        for theta in range(0,360):
            rad = (theta/180)*pi
            x= int(np.floor(gama*cos(rad)))
            y= int(np.floor(gama*sin(rad)))
            
            x += x0
            y=y0 - y
            #print(x,y)
            x = min(x,shape.shape[1]-1)
            y = min(y,shape.shape[0]-1)

            Fgama+=shape[y][x]
        CP.append(Fgama)

    for theta in range(0,360):
        Ftheta =0
        for gama in range(0,R):
            rad = (theta/180)*pi
            x= int(np.floor(gama*cos(rad)))
            y= int(np.floor(gama*sin(rad)))
            x+=x0
            y=y0 - y
            x = min(x,shape.shape[1]-1)
            y = min(y,shape.shape[0]-1)
            Ftheta+=shape[y][x]
        PR.append(Ftheta)
    

    fig1 = plt.figure()
    ax1 = plt.axes()

    x = list(range(0,R))
    ax1.plot(x, CP)
    plt.savefig('CP.png')
    # plt.show()
    fig2 = plt.figure()
    ax2 = plt.axes()
    x = list(range(0,360))
    ax2.plot(x, PR)
    plt.savefig('PR.png')
    # plt.show()

    



############################Test############################################
img = cv2.imread("0210.jpg", cv2.IMREAD_GRAYSCALE)
# shadowFreeImg = RemoveShadow(img,True)
binarizedImg =  Binarize_Histogram(img,"0210")
shapes = Diacritics_Segmentation(binarizedImg)
extractFeatures_CPRP(shapes[2])


