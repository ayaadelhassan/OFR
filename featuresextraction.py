import numpy as np
import cv2
from skimage.feature import hog as hogg
from preprocessing import *
from matplotlib import pyplot as plt
import math


def dist(x, y, x1, y1):
   return ((x - x1) ** 2 + (y - y1) ** 2) ** (0.5)

def slope(x, y, x1, y1):
   if y1 != y:
      return ((x1 - x) / (y1 - y))
   else :
      return 0

def getHVSL(edge_image, img, name=""):
    fld = cv2.ximgproc.createFastLineDetector()
    edge_image = np.uint8(edge_image)
    edge_image = Binarize_Histogram(edge_image)
    # histr = cv2.calcHist([edge_image],[0],None,[256],[0,256])
    # # show the plotting graph of an image
    # plt.plot(histr)
    # plt.show()
    lines = fld.detect(edge_image)
    no_of_horizontal_lines = 0.0
    no_of_vertical_lines = 0.0
    for line in lines:
        x0 = int(round(line[0][0]))
        y0 = int(round(line[0][1]))
        x1 = int(round(line[0][2]))
        y1 = int(round(line[0][3]))
        d = dist(x0, y0, x1, y1)
        # if d >= 0: #You can adjust the distance
        if np.abs(x0 - x1) == 0:
            no_of_vertical_lines += 1 
            cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 1, cv2.LINE_AA)
        if np.abs(y0 - y1) == 0:
            no_of_horizontal_lines += 1
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
            
    white_pix = cv2.countNonZero(edge_image)
    tot_pix = edge_image.size
    black_pix = tot_pix - white_pix
    # print(white_pix)
    # print(black_pix)
    # print(tot_pix)
    # print(edge_image.shape[0]*edge_image.shape[1])

    feature1 = (no_of_vertical_lines+no_of_horizontal_lines)/len(lines)
    feature2 = (feature1)-(black_pix/float(tot_pix))
    f_list = [feature1, feature2]
    f_list = np.array(f_list)
    return f_list


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


def getLVL(skeleton_image, img, name=""):
    fld = cv2.ximgproc.createFastLineDetector() #to get lines
    skeleton_image = np.uint8(skeleton_image)
    lines = fld.detect(skeleton_image)
    #features:
    no_of_vertical_lines = 0.0
    text_height = 0.0
    max_length = 0
    variance=0
    lines_mod = []
    for line in lines:
        x0 = int(round(line[0][0]))
        y0 = int(round(line[0][1]))
        x1 = int(round(line[0][2]))
        y1 = int(round(line[0][3]))
        if np.abs(x0 - x1) == 0:
            no_of_vertical_lines += 1 
            lines_mod.append(line[0])
            if(np.abs(y1-y0))>max_length:
                max_length = np.abs(y1-y0)

    lines_mod = np.asarray(lines_mod)
    nz = np.argwhere(skeleton_image < 255)
    yf,xf = nz[0] #fisrt black pixel
    yl,xl = nz[-1] #last black pixel
    text_height = np.abs(yl-yf)
    white_pix = cv2.countNonZero(skeleton_image)
    tot_pix = skeleton_image.size
    black_pix = tot_pix - white_pix
    if max_length == 0:
        max_length = 0.0001
    difference_ratio = text_height / max_length
    variance = np.var(lines_mod)
    if math.isnan(variance):
        variance = 0
    features = [text_height, no_of_vertical_lines, max_length,difference_ratio,variance]
    features = np.asarray(features)
    return features
   