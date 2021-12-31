import numpy as np
from preprocessing import *
from featuresextraction import *
import cv2
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


HVSL_feature = []
LVL_features = []
HPP_features = []
labels = []
feature_to_plot = []
for i in range(1, 10):
    input_dir = f'ACdata_base/{i}/'
    dirs = os.listdir(input_dir)
    for idx,img_dir in enumerate(dirs):
        #print("processing img "+str(img_dir))
        pre = img_dir.split('.')[0]
        img_dir = input_dir + img_dir
        img_rgb = cv2.imread(img_dir)
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        binarizedImg =  Binarize_Histogram(img,pre)

        ####preprocessing for feature extraction:
        ####skeletonization
        skeletonized = Skeletonization(binarizedImg, pre)
        ####edge detection:
        edged = LaplacianEdge(binarizedImg, pre)
        
        LVL_descriptor = getLVL(skeletonized,img,pre)
        HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
        HPP_descriptor = getHPP(cropToText(binarizedImg),pre)
        HVSL_feature.append(HVSL_decriptor)
        LVL_features.append(LVL_descriptor)
        HPP_features.append(HPP_descriptor)
        labels.append([i])

#######classifier

clf = svm.SVC()
#clf = DecisionTreeClassifier()
labels = np.array(labels)    
data_frame = np.array(LVL_features)#np.hstack((labels, HVSL_feature))

X_train, X_test, y_train, y_test = train_test_split(data_frame, labels, test_size=0.3,random_state=4, stratify=labels)

#tree = clf.fit(x_train,y_train) 
clf.fit(X_train,y_train)

predicted = clf.predict(X_test)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print('\n')
# print(classification_report(y_test, y_pred))
print(
    f"Classification report for classifier {clf}:\n"
    f"{classification_report(y_test, predicted)}\n"
)