import numpy as np
from preprocessing import *
import cv2
import os

input_dir = 'ACdata_base/5/'
dirs = os.listdir(input_dir)
for idx,img_dir in enumerate(dirs):
    print("processing img "+str(img_dir))
    pre = img_dir.split('.')[0]
    img_dir = input_dir + img_dir

    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # shadowFreeImg = RemoveShadow(img,True)
    binarizedImg =  Binarize_Histogram(img,pre)