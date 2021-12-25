import numpy as np
import cv2
from skimage.feature import hog as hogg

def dist(x, y, x1, y1):
   return ((x - x1) ** 2 + (y - y1) ** 2) ** (0.5)

def slope(x, y, x1, y1):
   if y1 != y:
      return ((x1 - x) / (y1 - y))
   else :
      return 0

def getHVSL(edge_image, img, name=""):
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(edge_image.astype('uint8'))
    no_of_horizontal_lines = 0.0
    no_of_vertical_lines = 0.0
    for line in lines:
        x0 = int(round(line[0][0]))
        y0 = int(round(line[0][1]))
        x1 = int(round(line[0][2]))
        y1 = int(round(line[0][3]))
        d = dist(x0, y0, x1, y1)
        if d > 10: #You can adjust the distance
            if np.abs(x0 - x1) >= 0 and np.abs(x0 - x1) <= 3:
                no_of_vertical_lines += 1 
                cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 1, cv2.LINE_AA)
            if np.abs(y0 - y1) >= 0 and np.abs(y0 - y1) <= 3:
                no_of_horizontal_lines += 1
                cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
    if no_of_horizontal_lines == 0:
        no_of_horizontal_lines = 1
    return no_of_vertical_lines,no_of_horizontal_lines


def getToE(edge_image, img, name=""):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(edge_image.astype('uint8'),winStride,padding,locations)
    return hist


def getToS(skel_image, img, name=""):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(skel_image.astype('uint8'),winStride,padding,locations)
    return hist