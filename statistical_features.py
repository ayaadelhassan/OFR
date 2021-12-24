import numpy as np
from preprocessing import *
from math import sin,cos,pi
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

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


############################Test############################################

def get_Features(binarized):
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

    features = [referenceline/hImg,blackwhiteRatioTotal,blackwhiteRatioAbove,blackwhiteRatioBelow,orientation,DenistyAbove,DenistyBelow]
    return features

base_dir='ACdata_base/'

fonts = os.listdir(base_dir)
X=[]
Y=[]
for font in fonts:
    data = os.listdir(base_dir+font)
    print("curFont",font)
    for img in data:
        img_dir = base_dir+font+'/'+img
        imgGray = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        binarizedImg =  Binarize_Histogram(imgGray,img_dir)
        features = get_Features(binarizedImg)
        X.append(features)
        Y.append(int(font))
    
X = np.array(X)
Y = np.array(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4, stratify=Y)
print("training size:",x_train.shape)
clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5), random_state=1,solver='lbfgs')
clf = clf.fit(x_train, y_train)

correct_pred = 0
for x,y in zip(x_test,y_test):
    y_pred = clf.predict(x)
    correct_pred += (y==y_pred)*1

print("accuracy: ",(correct_pred/len(y_test))*100)


from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def reduce_features(featureset):
    Xnew=[]
    for x in X:
        Xnew.append(x[featureset])
    return Xnew


features_x = list(range(7))
features_powerset = powerset(features_x)
accuracies=[]
for featureset in features_powerset:
    accuracy_set=0
    if len(featureset)>0:
        newX = reduce_features(featureset)
        