
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import os
import numpy as np



base_dir='ACdata_base/'

fonts = os.listdir(base_dir)

for font in fonts:
    h,w = 0,0
    data = os.listdir(base_dir+font)
    print("curFont",font)
    for img in data:
        img_dir = base_dir+font+'/'+img
        imgGray = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        resized_img = resize(imgGray, (200, 400))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
