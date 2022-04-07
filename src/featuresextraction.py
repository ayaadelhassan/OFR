import numpy as np
import cv2
from preprocessing import *
import math
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
import cv2
from skimage.transform import resize
import numpy as np
from preprocessing import *

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
   
def getHPP(whole_text, name=""):
    # horizontal projection
    h_proj = np.sum(whole_text == 0, axis=1)
    h_proj = h_proj / np.max(h_proj)

    return np.histogram(h_proj,bins=30)[0]
# ----------------------------------------------------------------------------------------------------------------------
def getTOS(imgGray):
    resized_img = resize(imgGray, (110, 200))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    return fd

def get_Reference_Line(binarized):
    hist = np.sum(1-binarized/255,axis=1)
    referenceline = np.argmax(hist)
    return referenceline

def get_BlackWhiteRatio(binarized):
    hImg, wImg = binarized.shape
    blackCount = np.sum(binarized==0)
    whiteCount = max(1,np.sum(binarized==255))
    return blackCount/whiteCount

def get_Components(binarized):
    contours,_ = cv2.findContours(255-binarized, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    return contours,len(contours)

def get_CountContours(contours,referenceline):
    countAbove,countBelow=0,0
    for cnt in contours:
        x,y,w,h  = cv2.boundingRect(cnt)
        if y+h <= referenceline:countAbove+=1
        if y > referenceline:countBelow+=1
        
    return countAbove,countBelow

def dist(x, y, x1, y1):
    return ((x - x1)**2)**(0.5) + ((y - y1)**2) ** (0.5)

def get_Orientation(contours,binarized):
    hImg, wImg = binarized.shape
    test = np.zeros((hImg,wImg,3))
    test[:,:,0] = binarized
    test[:,:,1] = binarized
    test[:,:,2] = binarized
    anglesSum = 0
    for cnt in contours:
        if cnt.shape[0] > 5:
            x,y,_,_  = cv2.boundingRect(cnt)
            ellipse = cv2.fitEllipse(cnt)
            (xc,yc),(d1,d2),angle = ellipse
            angle = 90 - angle
            anglesSum += angle
            cv2.putText(test,f'{int(angle)}',(x,y),0,0.25,(0,0,255)) 
            cv2.ellipse(test, ellipse, (255,0, 255), 1, cv2.LINE_AA)
    anglesMean = anglesSum/len(contours)
    cv2.imwrite("withEllipses.png", test)
    return anglesMean


def getStatisticalHVSL(edge_image, img=None, name=""):
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
                #cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 1, cv2.LINE_AA)
            if np.abs(y0 - y1) >= 0 and np.abs(y0 - y1) <= 3:
                no_of_horizontal_lines += 1
                #cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
    if no_of_horizontal_lines == 0:
        no_of_horizontal_lines = 1
    return no_of_vertical_lines,no_of_horizontal_lines



def getStatsFeatures(binarized):
    hImg, wImg = binarized.shape

    referenceline = get_Reference_Line(binarized)
    blackwhiteRatioTotal = get_BlackWhiteRatio(binarized)

    imgAboveRef = binarized[:referenceline,:]
    imgBelowRef = binarized[referenceline:,:]

    blackwhiteRatioAbove = get_BlackWhiteRatio(imgAboveRef)
    blackwhiteRatioBelow = get_BlackWhiteRatio(imgBelowRef)

    contoursTotal, contoursTotalCount = get_Components(binarized)

    contoursAboveCount,contoursBelowCount = get_CountContours(contoursTotal,referenceline)
    DenistyAbove = contoursAboveCount/contoursTotalCount
    DenistyBelow = contoursBelowCount/contoursTotalCount
    orientation = get_Orientation(contoursTotal,binarized)
    
    #########aya
    edges = LaplacianEdge(binarized)
    
    verticalCount,horizontalCount = getStatisticalHVSL(edges)
    

    features = [referenceline/hImg,blackwhiteRatioTotal,blackwhiteRatioAbove,blackwhiteRatioBelow,orientation,DenistyAbove,DenistyBelow,verticalCount,horizontalCount]
    return features

def dist(x, y, x1, y1):
    return ((x - x1)**2)**(0.5) + ((y - y1)**2) ** (0.5)

def getThicknessHist(skeleton,binarizedImg):
    blackPixelsSk = np.where(skeleton==0)
    contours,_ = cv2.findContours(255-binarizedImg, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    boundindBoxes = [cv2.boundingRect(contour) for contour in contours]
    thick_arr = []
    for i in range(0,len(blackPixelsSk[0]),3):
        y,x = blackPixelsSk[0][i],blackPixelsSk[1][i]

        for idx,(xB,yB,wB,hB) in enumerate(boundindBoxes):
            if y>=yB and x>=xB and y<=(yB+hB) and x<=(xB+wB):
                insideContour = contours[idx] 
                break

        insideContour = insideContour.T

        cols= insideContour[0][0]
        rows= insideContour[1][0]
        idxs = list(range(len(rows)))
        abovePoints = []
        belowPoints = []
        rightPoints = []
        leftPoints =  []
        
        for i,r,c in zip(idxs,rows,cols):
            if r<y and c==x:
                abovePoints.append(i)
            elif  r>y and c==x:
                belowPoints.append(i)
            elif r==y and c>x:
                rightPoints.append(i)
            elif r==y and c<x:
                leftPoints.append(i)
                
        if len(belowPoints):
            minIdx = np.argmin([dist(x,y,cols[belowPoints[i]],rows[belowPoints[i]]) for i in range(len(belowPoints))])
            nearBelow = rows[belowPoints[minIdx]],cols[belowPoints[minIdx]] 
        else: nearBelow = y,x

        if len(abovePoints):
            minIdx = np.argmin([dist(x,y,cols[abovePoints[i]],rows[abovePoints[i]]) for i in range(len(abovePoints))])
            nearAbove = rows[abovePoints[minIdx]],cols[abovePoints[minIdx]] 
        else: nearAbove = y,x

        if len(rightPoints): 
            minIdx = np.argmin([dist(x,y,cols[rightPoints[i]],rows[rightPoints[i]]) for i in range(len(rightPoints))])
            nearRight = rows[rightPoints[minIdx]],cols[rightPoints[minIdx]] 
        else: nearRight = y,x

        if len(leftPoints): 
            minIdx = np.argmin([dist(x,y,cols[leftPoints[i]],rows[leftPoints[i]]) for i in range(len(leftPoints))])
            nearLeft = rows[leftPoints[minIdx]],cols[leftPoints[minIdx]] 
        else: nearLeft = y,x
            
        distVer = dist(nearBelow[1],nearBelow[0],nearAbove[1],nearAbove[0])
        distHor = dist(nearRight[1],nearRight[0],nearLeft[1],nearLeft[0])
        thickness = min(distVer,distHor)
        thick_arr.append(thickness)
    thick_hist,bins = np.histogram(thick_arr, 10)
    print(thick_hist,bins)
    return list(thick_hist)+list(bins)