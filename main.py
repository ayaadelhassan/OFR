import numpy as np
from preprocessing import *
from featuresextraction import *
import cv2
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from scipy.stats import mode


HVSL_feature = []
LVL_features = []
HPP_features = []
TOS_features = []
Stats_features = []
TH_features = []
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
        ####our skeletonization
        skeletonized = Skeletonization(binarizedImg, pre)

        ###their skeletonization
        skeleton = 255 - skeletonize(1-binarizedImg/255)*255
        ####edge detection:
        edged = LaplacianEdge(binarizedImg, pre)
        
        LVL_descriptor = getLVL(skeletonized,img,pre)
        HVSL_decriptor = getHVSL(edged,img_rgb,pre) #contains 2 features (V/H,(#of black pixels/((H/total)+(V/total))))
        HPP_descriptor = getHPP(cropToText(binarizedImg),pre)
        TOS_descriptor = getTOS(img)
        Stats_descriptor = getStatsFeatures(binarizedImg)
        #TH_descriptor = getThicknessHist(skeleton, binarizedImg)
        HVSL_feature.append(HVSL_decriptor)
        LVL_features.append(LVL_descriptor)
        HPP_features.append(HPP_descriptor)
        TOS_features.append(TOS_descriptor)
        Stats_features.append(Stats_descriptor)
        #TH_features.append(TH_descriptor)
        labels.append([i])

#######classifier
labels = np.array(labels)    


X_train, X_test, y_train, y_test = train_test_split(HVSL_feature, labels, test_size=0.3,random_state=4, stratify=labels)
clf_HVSL = svm.SVC()
clf_HVSL.fit(X_train, y_train)
predicted_HVSL = clf_HVSL.predict(X_test)

X_train, X_test, y_train, y_test = train_test_split(HPP_features, labels, test_size=0.3,random_state=4, stratify=labels)
clf_HPP = svm.SVC()
clf_HPP.fit(X_train, y_train)
predicted_HPP = clf_HPP.predict(X_test)

X_train, X_test, y_train, y_test = train_test_split(Stats_features, labels, test_size=0.3,random_state=4, stratify=labels)
clf_Stats = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(9,18),random_state=1,solver='lbfgs',max_iter=10000)
clf_Stats.fit(X_train,y_train)
predicted_Stats = clf_Stats.predict(X_test)

X_train, X_test, y_train, y_test = train_test_split(TOS_features, labels, test_size=0.3,random_state=4, stratify=labels)
clf_TOS = svm.SVC()
clf_TOS.fit(X_train,y_train)
predicted_TOS = clf_TOS.predict(X_test)

X_train, X_test, y_train, y_test = train_test_split(LVL_features, labels, test_size=0.3,random_state=4, stratify=labels)
clf_LVL = svm.SVC()
clf_LVL.fit(X_train,y_train)
predicted_LVL = clf_LVL.predict(X_test)

stacked = np.vstack((predicted_Stats,predicted_TOS,predicted_HPP, predicted_HVSL, predicted_LVL))
predicted = mode(stacked)[0][0]
print(predicted)

# predicted = mode().T,axis=0)
# print(predicted_HVSL.shape)
# print(predicted)
# print(y_test.shape)
#voting_clf = VotingClassifier(estimators=[('clf_HVSL', clf_HVSL), ('clf_Stats', clf_Stats)], voting='hard')
#voting_clf.fit(X_train, y_train)
#predicted = voting_clf.predict(X_test)

# acc = accuracy_score(y_test, preds)
# f1 = f1_score(y_test, preds, average='weighted')
# print("Accuracy is: " + str(acc))
# print("F1 Score is: " + str(f1))
# predicted = clf.predict(X_test)
print(
    #f"Classification report for classifier {voting_clf}:\n"
    f"{classification_report(y_test, predicted)}\n"
)
