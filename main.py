import numpy as np
from preprocessing import *
from featuresextraction import *
import cv2
import os
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score


input_dir = 'ACdata_base/1/'
dirs = os.listdir(input_dir)
hog_features = []
hog_features2 = []
HVSL_feature = []
labels = []
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["1"])

input_dir = 'ACdata_base/2/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["2"])

input_dir = 'ACdata_base/3/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["3"])

input_dir = 'ACdata_base/4/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["4"])

input_dir = 'ACdata_base/5/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["5"])

input_dir = 'ACdata_base/6/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["6"])

input_dir = 'ACdata_base/7/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["7"])

input_dir = 'ACdata_base/8/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["8"])

input_dir = 'ACdata_base/9/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir
    img_rgb = cv2.imread(img_dir)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)
    ####preprocessing for feature extraction:
    ####skeletonization
    skeletonized = Skeletonization(binarizedImg, pre)
    ####edge detection:
    edged = LaplacianEdge(binarizedImg, pre)
    HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
    ToE_descriptor = getToE(edged,img,pre) #histogram
    ToS_descriptor = getToS(skeletonized,img,pre) #histogram
    hog_features.append(ToS_descriptor)
    hog_features2.append(ToE_descriptor)
    HVSL_feature.append([HVSL_decriptor[0], HVSL_decriptor[1]])
    labels.append(["9"])


print(len(hog_features))
print(len(hog_features2))
print(len(HVSL_feature))
print(len(labels))
print(HVSL_feature)
#######classifier

clf = svm.SVC()
hog_features = np.array(hog_features)
labels = np.array(labels)    
data_frame = np.hstack((labels,HVSL_feature))
np.random.shuffle(data_frame)
percentage = 80
partition = int(len(hog_features)*percentage/100)

x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))
